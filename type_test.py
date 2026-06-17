import re
from akasha_db import init_db, log_event, get_net_hydration_today

init_db()

# (module, feature, hydration_factor)
# hydration_factor = fraction of volume that counts toward net hydration
# None = not a beverage, hydration calc not applicable
RULES = {
    "water": ("body", "hydration", 1.0),
    "hydrate": ("body", "hydration", 1.0),
    "h2o": ("body", "hydration", 1.0),
    "coffee": ("body", "caffeine", 0.95),
    "tea": ("body", "caffeine", 0.97),
    "espresso": ("body", "caffeine", 0.9),
    "beer": ("body", "drinks", 0.85),
    "wine": ("body", "drinks", 0.55),
    "vodka": ("body", "drinks", 0.25),
    "whiskey": ("body", "drinks", 0.25),
    "soda": ("body", "drinks", 0.95),
    "sprite": ("body", "drinks", 0.95),
    "coke": ("body", "drinks", 0.95),
    "pepsi": ("body", "drinks", 0.95),
    "fanta": ("body", "drinks", 0.95),
    "mountain dew": ("body", "drinks", 0.95),
    "gatorade": ("body", "drinks", 1.0),
    "lemonade": ("body", "drinks", 0.95),
    "juice": ("body", "drinks", 0.95),
    "cocktail": ("body", "drinks", 0.45),
    "alcohol": ("body", "drinks", 0.5),
    "drank": ("body", "drinks", 0.8),
    "workout": ("body", "training", None),
    "lifted": ("body", "training", None),
    "ran": ("body", "training", None),
    "gym": ("body", "training", None),
    "exercise": ("body", "training", None),
    "jog": ("body", "training", None),
    "swam": ("body", "training", None),
    "cardio": ("body", "training", None),
    "stretched": ("body", "training", None),
    "vitamin": ("body", "supplement_log", None),
    "supplement": ("body", "supplement_log", None),
    "took my": ("body", "supplement_log", None),
    "multivitamin": ("body", "supplement_log", None),
    "fish oil": ("body", "supplement_log", None),
    "protein": ("body", "supplement_log", None),
    "creatine": ("body", "supplement_log", None),
}

UNIT_PATTERN = re.compile(
    r"(\d+\.?\d*)\s*(ounces?|oz|cups?|ml|milliliters?|liters?|l|glasses?)",
    re.IGNORECASE
)

ML_CONVERSIONS = {
    "ounce": 29.5735, "ounces": 29.5735, "oz": 29.5735,
    "cup": 236.588, "cups": 236.588,
    "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0,
    "liter": 1000.0, "liters": 1000.0, "l": 1000.0,
    "glass": 240.0, "glasses": 240.0,
}

def extract_amount(text):
    match = UNIT_PATTERN.search(text.lower())
    if match:
        amount = float(match.group(1))
        unit = match.group(2).lower()
        ml = amount * ML_CONVERSIONS.get(unit, 0)
        return amount, unit, ml
    return None, None, None

print("Type a sentence and hit enter. Type 'quit' to stop.\n")

while True:
    text = input("> ").strip()
    if text.lower() == "quit":
        break
    if not text:
        continue

    matched = False
    for keyword, (module, feature, hydration_factor) in RULES.items():
        if keyword in text.lower():
            amount, unit, ml = extract_amount(text)
            log_event(module, feature, text, amount, unit, ml, hydration_factor)

            if amount and hydration_factor is not None:
                net = ml * hydration_factor
                print(f"Logged to {module}/{feature} ({amount} {unit} = {net:.0f}ml net hydration).")
                print(f"Net hydration today: {get_net_hydration_today():.0f}ml")
            elif amount:
                print(f"Logged to {module}/{feature} ({amount} {unit}, no hydration factor).")
            else:
                print(f"Logged to {module}/{feature} (no quantity detected).")
            matched = True
            break

    if not matched:
        print("No match.")
