from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import openai
import requests
import os
from dotenv import load_dotenv
import logging
import json
import asyncio
import aiohttp
import re

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Debugging: Check if .env is loading correctly
logging.info("Loaded OpenAI Key: %s", os.getenv("OPENAI_API_KEY"))
logging.info("Loaded USDA Key: %s", os.getenv("USDA_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Fetch API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USDA_API_KEY = os.getenv("USDA_API_KEY")

# Debugging: Check if API keys are loaded
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing. Make sure it's in your .env file!")
if not USDA_API_KEY:
    raise ValueError("USDA_API_KEY is missing. Make sure it's in your .env file!")

# USDA Food Database API URL
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"

# Define request model
class MealPlanRequest(BaseModel):
    dietary_preferences: str  # e.g., vegan, keto, balanced
    gut_health_score: str  # e.g., "poor", "moderate", "good"
    fitness_goal: str  # e.g., "muscle gain", "weight loss"
    day: int = Query(1, ge=1, le=7, description="Choose a day from 1 to 7")  # Fetch one day at a time

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the Gut Health AI API! Use /docs to explore the endpoints."}

# Function to fetch key nutritional data asynchronously
async def fetch_nutritional_info(session, meal):
    params = {"api_key": USDA_API_KEY, "query": meal, "dataType": "Foundation"}
    async with session.get(USDA_API_URL, params=params) as response:
        data = await response.json()
        
        # Extract only important nutritional information
        nutrients = {}
        if "foods" in data and len(data["foods"]) > 0:
            food = data["foods"][0]
            for nutrient in food.get("foodNutrients", []):
                if nutrient.get("nutrientName") in ["Energy", "Protein", "Fiber, total dietary", "Probiotics"]:
                    nutrients[nutrient.get("nutrientName")] = nutrient.get("value")
        
        return meal, nutrients

# Generate AI-powered meal plan and fetch nutritional info
@app.post("/generate_meal_plan")
async def generate_meal_plan(request: MealPlanRequest):
    try:
        # Define AI prompt for OpenAI
        prompt = (f"Generate a gut-friendly meal plan for one day (Day {request.day}) of a {request.dietary_preferences} diet. "
                  f"Consider the user's gut health score: {request.gut_health_score}. "
                  f"The fitness goal is {request.fitness_goal}. "
                  f"Include probiotic and prebiotic foods where appropriate. "
                  f"Respond ONLY with a valid JSON array of meal names. No extra text, no explanations, no formatting hints.")
        
        # OpenAI API Call (Updated for openai>=1.0.0)
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are a gut-health nutrition expert. Always respond with valid JSON."},
                          {"role": "user", "content": prompt}]
            )
            
            # Validate and clean OpenAI response
            raw_response = response.choices[0].message.content.strip()
            logging.info("Raw OpenAI Response: %s", raw_response)  # Log response for debugging
            
            # Extract JSON block using regex (handles extra text issues)
            json_match = re.search(r'\[.*?\]', raw_response, re.DOTALL)
            if json_match:
                raw_response = json_match.group(0)
            
            try:
                meal_list = json.loads(raw_response)
                if not isinstance(meal_list, list):
                    raise ValueError("OpenAI response is not a valid JSON list.")
            except json.JSONDecodeError as e:
                logging.error("Failed to parse OpenAI response as JSON: %s", raw_response)
                raise HTTPException(status_code=500, detail=f"OpenAI response is not in expected JSON format: {str(e)}")
            
            # Fetch nutritional information for each meal asynchronously
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_nutritional_info(session, meal) for meal in meal_list]
                results = await asyncio.gather(*tasks)
                nutritional_info = {meal: data for meal, data in results}
            
            return {"day": request.day, "meal_plan": meal_list, "nutritional_info": nutritional_info}
        except openai.OpenAIError as e:
            logging.error("OpenAI API Error: %s", e)
            if "insufficient_quota" in str(e):
                return {"error": "OpenAI API quota exceeded. Please check your OpenAI account or upgrade your plan."}
            else:
                raise e
    except Exception as e:
        logging.error("Unexpected Error: %s", e)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Run the app (if running locally)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
