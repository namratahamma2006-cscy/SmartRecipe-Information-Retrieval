"""SmartRecipe - a recipe information retrieval system."""

from .engine import SmartRecipeEngine, load_recipes
from .query import parse_query

__all__ = ["SmartRecipeEngine", "load_recipes", "parse_query"]
__version__ = "1.0.0"
