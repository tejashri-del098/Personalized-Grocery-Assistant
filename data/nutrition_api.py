"""
nutrition_api.py — OpenFoodFacts API Client

Fetches nutritional information for grocery products using the free
OpenFoodFacts REST API (https://world.openfoodfacts.org).

Response data includes:
  • Product name and brand
  • Ingredients list
  • Allergens
  • Nutri-Score grade (A–E)
  • Key nutriments (energy, fat, sugars, salt, proteins, fiber)
"""

import requests

# ── API Configuration ────────────────────────────────────────────────
BASE_URL = "https://world.openfoodfacts.org/api/v0/product"
TIMEOUT = 10  # seconds
USER_AGENT = "PersonalizedGroceryAssistant/1.0"

# ── Fallback mock data for testing/demos ─────────────────────────────
MOCK_DATABASE = {
    "3017620422003": {
        "barcode": "3017620422003",
        "name": "Nutella",
        "brand": "Ferrero",
        "ingredients": "Sugar, Palm oil, Hazelnuts 13%, Skimmed milk powder 8.7%, Fat-reduced cocoa 7.4%, Emulsifier: lecithin (soya), Vanillin",
        "allergens": "en:nuts, en:milk, en:soybeans",
        "nutriscore": "E",
        "nutriments": {
            "energy_kcal": 539.0,
            "fat_g": 30.9,
            "saturated_fat_g": 10.6,
            "sugars_g": 56.3,
            "salt_g": 0.11,
            "proteins_g": 6.3,
            "fiber_g": 3.4,
        },
        "image_url": "",
    }
}


def fetch_nutrition(barcode: str) -> dict | None:
    """Fetch nutritional data for a product by barcode.

    Args:
        barcode: EAN-13, UPC-A, or similar barcode string.

    Returns:
        A dict with structured nutritional info, or ``None`` if the
        product is not found.

        Example return value::

            {
                "barcode": "3017620422003",
                "name": "Nutella",
                "brand": "Ferrero",
                "ingredients": "Sugar, Palm oil, Hazelnuts ...",
                "allergens": "en:nuts, en:milk, en:soybeans",
                "nutriscore": "E",
                "nutriments": {
                    "energy_kcal": 539,
                    "fat_g": 30.9,
                    "saturated_fat_g": 10.6,
                    "sugars_g": 56.3,
                    "salt_g": 0.107,
                    "proteins_g": 6.3,
                    "fiber_g": 3.4,
                },
                "image_url": "https://...",
            }
    """
    url = f"{BASE_URL}/{barcode}.json"
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        # Check against mock database if API fails (e.g. rate limit / 429 error)
        if barcode in MOCK_DATABASE:
            print(f"  [!] OpenFoodFacts error ({exc})")
            print(f"  [!] Using offline fallback data for sample barcode {barcode}.")
            return MOCK_DATABASE[barcode]
            
        print(f"  [!] Network error fetching barcode {barcode}: {exc}")
        return None

    data = response.json()

    # API returns status==0 when the product is not in the database
    if data.get("status") != 1:
        return None

    product = data.get("product", {})
    nutriments = product.get("nutriments", {})

    return {
        "barcode": barcode,
        "name": product.get("product_name", "Unknown Product"),
        "brand": product.get("brands", "Unknown Brand"),
        "ingredients": product.get("ingredients_text", "N/A"),
        "allergens": product.get("allergens", "None listed"),
        "nutriscore": product.get("nutriscore_grade", "N/A").upper(),
        "nutriments": {
            "energy_kcal": _safe_float(nutriments.get("energy-kcal_100g")),
            "fat_g": _safe_float(nutriments.get("fat_100g")),
            "saturated_fat_g": _safe_float(nutriments.get("saturated-fat_100g")),
            "sugars_g": _safe_float(nutriments.get("sugars_100g")),
            "salt_g": _safe_float(nutriments.get("salt_100g")),
            "proteins_g": _safe_float(nutriments.get("proteins_100g")),
            "fiber_g": _safe_float(nutriments.get("fiber_100g")),
        },
        "image_url": product.get("image_url", ""),
    }


def search_product(query: str, page_size: int = 5) -> list[dict]:
    """Search OpenFoodFacts by product name.

    Useful as a fallback when a barcode is not available but the
    product label has been identified via object detection.

    Args:
        query: Free-text search term (e.g. ``"banana"``).
        page_size: Number of results to return.

    Returns:
        List of simplified product dicts.
    """
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": query,
        "json": 1,
        "page_size": page_size,
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(url, params=params, headers=headers,
                                timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[!] Search error for '{query}': {exc}")
        return []

    data = response.json()
    products = data.get("products", [])

    results = []
    for p in products:
        results.append({
            "barcode": p.get("code", ""),
            "name": p.get("product_name", "Unknown"),
            "brand": p.get("brands", "Unknown"),
            "nutriscore": p.get("nutriscore_grade", "N/A").upper(),
        })

    return results


# ── Private helpers ──────────────────────────────────────────────────

def _safe_float(value) -> float:
    """Convert a value to float, returning 0.0 on failure."""
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0.0
