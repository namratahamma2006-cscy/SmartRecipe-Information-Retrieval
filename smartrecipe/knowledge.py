"""Domain knowledge for SmartRecipe: synonyms, ingredient families, dietary
categories, pantry staples, substitutions and query-concept expansions.

All keys and values are written in plain, human-readable form. They are passed
through the same text-preprocessing pipeline as recipes and queries, so
plurals and casing never matter ("Garbanzo Beans" == "garbanzo bean").
"""

# --------------------------------------------------------------------------
# Synonyms: regional / alternative name  ->  canonical ingredient name
# --------------------------------------------------------------------------
SYNONYMS = {
    # legumes
    "garbanzo": "chickpea",
    "garbanzo bean": "chickpea",
    "chick pea": "chickpea",
    "chana": "chickpea",
    "chole": "chickpea",
    "dal": "lentil",
    "dhal": "lentil",
    "daal": "lentil",
    "masoor": "lentil",
    "rajma": "kidney bean",
    # vegetables
    "aubergine": "eggplant",
    "brinjal": "eggplant",
    "baingan": "eggplant",
    "courgette": "zucchini",
    "capsicum": "bell pepper",
    "sweet pepper": "bell pepper",
    "coriander": "cilantro",
    "coriander leaf": "cilantro",
    "dhania": "cilantro",
    "scallion": "green onion",
    "spring onion": "green onion",
    "maize": "corn",
    "sweetcorn": "corn",
    "sweet corn": "corn",
    "aloo": "potato",
    "spud": "potato",
    "gobi": "cauliflower",
    "palak": "spinach",
    "matar": "pea",
    "beansprout": "bean sprout",
    "romaine": "lettuce",
    "cherry tomato": "tomato",
    "roma tomato": "tomato",
    "red onion": "onion",
    "shallot": "onion",
    "chilli": "chili",
    "chile": "chili",
    "green chili": "chili",
    "red chili": "chili",
    "jalapeno": "chili",
    "mirch": "chili",
    "chilli flake": "chili flakes",
    "chili flake": "chili flakes",
    "red pepper flake": "chili flakes",
    "chilli powder": "chili powder",
    "cayenne": "chili powder",
    "lemon juice": "lemon",
    "lime juice": "lime",
    "garlic clove": "garlic",
    # proteins
    "prawn": "shrimp",
    "king prawn": "shrimp",
    "chicken breast": "chicken",
    "chicken thigh": "chicken",
    "murgh": "chicken",
    "ground beef": "beef",
    "minced beef": "beef",
    "mince": "beef",
    "beef mince": "beef",
    "bean curd": "tofu",
    # dairy
    "curd": "yogurt",
    "yoghurt": "yogurt",
    "dahi": "yogurt",
    "heavy cream": "cream",
    "double cream": "cream",
    "whipping cream": "cream",
    "fresh cream": "cream",
    "cheddar cheese": "cheddar",
    "parmesan cheese": "parmesan",
    "parmigiano": "parmesan",
    "mozzarella cheese": "mozzarella",
    "feta cheese": "feta",
    "clarified butter": "ghee",
    # pantry
    "plain flour": "flour",
    "all purpose flour": "flour",
    "maida": "flour",
    "wheat flour": "flour",
    "caster sugar": "sugar",
    "white sugar": "sugar",
    "vegetable stock": "vegetable broth",
    "veg stock": "vegetable broth",
    "chicken stock": "chicken broth",
    "stock": "broth",
    "noodle": "noodles",
    "basmati": "basmati rice",
    "arborio": "arborio rice",
    "peanut": "peanut",
    "groundnut": "peanut",
    "oat": "oats",
    "rolled oat": "oats",
    "porridge oat": "oats",
    "cocoa": "cocoa powder",
    "sesame paste": "tahini",
    "pitta": "pita bread",
    "pita": "pita bread",
    "seaweed": "seaweed",
    "nori": "seaweed",
    "wakame": "seaweed",
    "ketchup": "tomato ketchup",
    "vanilla extract": "vanilla",
    "black pepper": "black pepper",
    "pepper": "black pepper",
}

# --------------------------------------------------------------------------
# Ingredient families: specific ingredient -> general family.
# Lets a user who has "pasta" match a recipe that needs "spaghetti" and a user
# who has "cheddar" match a recipe that just needs "cheese".
# --------------------------------------------------------------------------
FAMILIES = {
    "parmesan": "cheese",
    "mozzarella": "cheese",
    "feta": "cheese",
    "cheddar": "cheese",
    "paneer": "cheese",
    "spaghetti": "pasta",
    "macaroni": "pasta",
    "salmon": "fish",
    "tuna": "fish",
    "rice noodles": "noodles",
    "egg noodles": "noodles",
    "basmati rice": "rice",
    "arborio rice": "rice",
    "corn tortilla": "tortilla",
    "flour tortilla": "tortilla",
    "vegetable broth": "broth",
    "chicken broth": "broth",
    "kidney bean": "bean",
    "black bean": "bean",
    "olive oil": "oil",
    "sesame oil": "oil",
    "brown sugar": "sugar",
    "pita bread": "bread",
    "burger bun": "bread",
    "berry": "fruit",
    "chili flakes": "chili",
    "chili powder": "chili",
}

# --------------------------------------------------------------------------
# Dietary categories. An ingredient may belong to several.
# --------------------------------------------------------------------------
CATEGORIES = {
    "meat": ["chicken", "beef", "pork", "lamb", "bacon", "ham", "sausage"],
    "fish": ["fish", "salmon", "tuna", "anchovy", "fish sauce", "green curry paste"],
    "seafood": ["shrimp"],
    "dairy": [
        "milk", "butter", "cheese", "parmesan", "mozzarella", "feta", "cheddar",
        "cream", "yogurt", "paneer", "ghee", "chocolate chip",
    ],
    "egg": ["egg", "egg noodles"],
    "honey": ["honey"],
    "gluten": [
        "flour", "bread", "pasta", "spaghetti", "macaroni", "egg noodles",
        "soy sauce", "pita bread", "burger bun", "flour tortilla", "oats",
        "breadcrumb", "couscous",
    ],
    "nuts": ["peanut", "walnut", "almond", "cashew", "peanut butter"],
}

# Diet label -> categories that the diet forbids.
DIET_RULES = {
    "Vegetarian": ["meat", "fish", "seafood"],
    "Vegan": ["meat", "fish", "seafood", "dairy", "egg", "honey"],
    "Gluten-Free": ["gluten"],
    "Dairy-Free": ["dairy"],
    "Nut-Free": ["nuts"],
}

# Words that may appear in a query and the category they exclude
# ("no dairy", "without meat", "egg free").
CATEGORY_WORDS = {
    "meat": "meat", "dairy": "dairy", "egg": "egg", "gluten": "gluten",
    "nut": "nuts", "fish": "fish", "seafood": "seafood", "honey": "honey",
}

# --------------------------------------------------------------------------
# Pantry staples: almost every kitchen has them, so they are not reported as
# "missing" (user can switch this off).
# --------------------------------------------------------------------------
PANTRY_STAPLES = ["salt", "black pepper", "water", "oil"]

# --------------------------------------------------------------------------
# Substitutions for missing ingredients.
# --------------------------------------------------------------------------
SUBSTITUTES = {
    "butter": ["olive oil", "ghee", "coconut oil"],
    "cream": ["coconut milk", "yogurt", "milk"],
    "milk": ["oat milk", "coconut milk", "water"],
    "yogurt": ["sour cream", "coconut yogurt", "buttermilk"],
    "egg": ["flax egg (1 tbsp flax + 3 tbsp water)", "banana", "yogurt"],
    "paneer": ["tofu", "halloumi"],
    "chicken": ["tofu", "chickpea", "paneer"],
    "beef": ["lentil", "mushroom", "chicken"],
    "shrimp": ["chicken", "tofu"],
    "salmon": ["tuna", "white fish", "tofu"],
    "lemon": ["lime", "vinegar"],
    "lime": ["lemon", "vinegar"],
    "cilantro": ["parsley", "mint"],
    "parsley": ["cilantro", "basil"],
    "basil": ["oregano", "parsley", "spinach"],
    "mint": ["basil", "cilantro"],
    "soy sauce": ["tamari (gluten-free)", "coconut aminos"],
    "fish sauce": ["soy sauce", "salt"],
    "spaghetti": ["pasta", "zucchini noodles"],
    "macaroni": ["pasta"],
    "rice": ["quinoa", "cauliflower rice"],
    "basmati rice": ["rice"],
    "arborio rice": ["rice"],
    "honey": ["maple syrup", "sugar"],
    "sugar": ["honey", "maple syrup"],
    "vegetable broth": ["chicken broth", "water + stock cube"],
    "chicken broth": ["vegetable broth", "water + stock cube"],
    "garam masala": ["curry powder"],
    "parmesan": ["nutritional yeast", "cheddar"],
    "mozzarella": ["cheddar", "paneer"],
    "cheddar": ["mozzarella", "parmesan"],
    "feta": ["paneer", "goat cheese"],
    "ghee": ["butter", "oil"],
    "onion": ["leek", "green onion"],
    "green onion": ["onion"],
    "garlic": ["garlic powder"],
    "ginger": ["ground ginger"],
    "tomato": ["canned tomato", "tomato paste"],
    "coconut milk": ["cream", "milk"],
    "tahini": ["peanut butter"],
    "spinach": ["kale"],
    "bell pepper": ["zucchini", "carrot"],
    "mushroom": ["zucchini", "eggplant"],
    "chickpea": ["white bean", "kidney bean"],
    "black bean": ["kidney bean"],
    "kidney bean": ["black bean"],
    "corn tortilla": ["flour tortilla"],
    "flour tortilla": ["corn tortilla"],
    "cumin": ["garam masala"],
    "white wine": ["vegetable broth + lemon"],
    "chocolate chip": ["chopped chocolate"],
    "buttermilk": ["milk + lemon"],
    "berry": ["banana", "apple"],
    "walnut": ["pecan", "almond"],
    "quinoa": ["couscous", "rice"],
    "lentil": ["split pea"],
    "chili": ["chili powder", "chili flakes"],
}

# --------------------------------------------------------------------------
# Query concepts: descriptive words expanded into ingredient/dish terms.
# Expanded terms are added to the query at a reduced weight.
# --------------------------------------------------------------------------
CONCEPTS = {
    "spicy": ["chili", "chili powder", "chili flakes", "garam masala", "curry"],
    "hot": ["chili", "chili powder"],
    "sweet": ["sugar", "honey", "dessert"],
    "creamy": ["cream", "coconut milk", "cheese", "yogurt"],
    "cheesy": ["cheese", "cheddar", "mozzarella", "parmesan"],
    "healthy": ["salad", "vegetable", "quinoa", "lentil", "spinach"],
    "protein": ["chicken", "egg", "tofu", "lentil", "chickpea", "paneer"],
    "seafood": ["shrimp", "salmon", "fish"],
    "meaty": ["chicken", "beef"],
    "chocolate": ["cocoa powder", "chocolate chip"],
    "chocolatey": ["cocoa powder", "chocolate chip"],
    "veggie": ["vegetable", "vegetarian"],
    "fruity": ["fruit", "banana", "berry", "mango", "apple"],
    "comfort": ["cheese", "potato", "soup", "pasta"],
    "noodles": ["pasta", "spaghetti", "rice noodles"],
    "curry": ["garam masala", "coconut milk", "turmeric"],
}

# Course and cuisine words recognised as soft intents (used for ranking boost).
COURSE_WORDS = {
    "breakfast": "Breakfast", "brunch": "Breakfast", "morning": "Breakfast",
    "dessert": "Dessert", "sweet": "Dessert", "baking": "Dessert", "cake": "Dessert",
    "snack": "Snack", "appetizer": "Snack", "starter": "Snack", "dip": "Snack",
    "soup": "Soup", "salad": "Salad", "drink": "Drink", "beverage": "Drink",
    "smoothie": "Drink", "side": "Side", "dinner": "Main", "lunch": "Main",
    "main": "Main",
}

CUISINE_WORDS = {
    "indian": "Indian", "italian": "Italian", "mexican": "Mexican",
    "thai": "Thai", "chinese": "Chinese", "japanese": "Japanese",
    "greek": "Mediterranean", "mediterranean": "Mediterranean",
    "middle eastern": "Middle Eastern", "lebanese": "Middle Eastern",
    "american": "American", "asian": "Asian",
}
# Cuisine groups so "asian" matches Thai, Chinese and Japanese recipes.
CUISINE_GROUPS = {"Asian": ["Thai", "Chinese", "Japanese", "Indian"]}

DIET_WORDS = {
    "vegetarian": "Vegetarian", "veg": "Vegetarian", "veggie": "Vegetarian",
    "vegan": "Vegan", "plant based": "Vegan",
    "gluten free": "Gluten-Free", "glutenfree": "Gluten-Free", "celiac": "Gluten-Free",
    "dairy free": "Dairy-Free", "lactose free": "Dairy-Free",
    "nut free": "Nut-Free",
}

QUICK_WORDS = {"quick", "fast", "speedy", "rapid", "hurry", "instant"}
QUICK_MINUTES = 30

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by could did do does doing down during each
few for from further had has have having he her here hers him his how i if in
into is it its itself just me more most my myself nor now of off on once only or
other our ours out over own same she should so some such than that the their
them then there these they this those through to too under until up very was we
were what when where which while who whom why will with would you your yours
yourself
recipe recipes dish dishes food foods meal meals make making made want wanted
something anything like love please give show find get need cook cooking cooked
using use used idea ideas tonight today good best nice tasty delicious yummy
also can let lets try minute minutes min mins hour hours time
""".split())
