"""
formatter.py — Rich Console Output Formatting

Provides clean, structured terminal output for displaying:
  • Barcode & detection scan results
  • Nutritional information panels
  • Price comparison tables
  • Health summary with Nutri-Score visualization
"""


# ── ANSI colour codes ────────────────────────────────────────────────
class Colors:
    HEADER  = "\033[95m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"


# ── Nutri-Score colour map ───────────────────────────────────────────
NUTRI_COLORS = {
    "A": Colors.GREEN,
    "B": Colors.GREEN,
    "C": Colors.YELLOW,
    "D": Colors.YELLOW,
    "E": Colors.RED,
}


def _line(char: str = "─", width: int = 60) -> str:
    return char * width


def _header(title: str, width: int = 60) -> str:
    pad = width - len(title) - 4
    left = pad // 2
    right = pad - left
    return (
        f"\n{Colors.BOLD}{Colors.CYAN}"
        f"{'═' * left}  {title}  {'═' * right}"
        f"{Colors.RESET}\n"
    )


# ── Detection results ────────────────────────────────────────────────

def print_detection_summary(barcode_results: list[dict],
                            detection_results: list[dict]) -> None:
    """Print a summary of what was found in the image.

    Args:
        barcode_results: Output from barcode_scanner.
        detection_results: Output from object_detector.
    """
    print(_header("SCAN RESULTS"))

    if barcode_results:
        print(f"  {Colors.GREEN}▸ Barcodes detected:{Colors.RESET} "
              f"{len(barcode_results)}")
        for bc in barcode_results:
            print(f"    • [{bc['type']}] {Colors.BOLD}{bc['data']}{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}▸ No barcodes detected{Colors.RESET}")

    print()

    if detection_results:
        print(f"  {Colors.GREEN}▸ Products detected:{Colors.RESET} "
              f"{len(detection_results)}")
        for det in detection_results:
            conf = det['confidence'] * 100
            print(f"    • {Colors.BOLD}{det['label'].capitalize()}{Colors.RESET}"
                  f" ({conf:.1f}% confidence)")
    else:
        print(f"  {Colors.DIM}▸ No products detected via object recognition"
              f"{Colors.RESET}")

    print()


# ── Nutrition panel ──────────────────────────────────────────────────

def print_nutrition(product_info: dict) -> None:
    """Print a formatted nutrition facts panel.

    Args:
        product_info: Unified product dict from product_mapper.
    """
    nutrition = product_info.get("nutrition")
    name = product_info.get("name", "Unknown")

    print(_header(f"NUTRITION — {name}"))

    if not nutrition:
        print(f"  {Colors.DIM}No nutritional data available for this product."
              f"{Colors.RESET}\n")
        return

    # Product identity
    print(f"  {Colors.BOLD}Product:{Colors.RESET}  {nutrition.get('name', 'N/A')}")
    print(f"  {Colors.BOLD}Brand:{Colors.RESET}    {nutrition.get('brand', 'N/A')}")
    print(f"  {Colors.BOLD}Barcode:{Colors.RESET}  {nutrition.get('barcode', 'N/A')}")
    print()

    # Nutri-Score
    score = nutrition.get("nutriscore", "N/A")
    color = NUTRI_COLORS.get(score, Colors.DIM)
    print(f"  {Colors.BOLD}Nutri-Score:{Colors.RESET}  "
          f"{color}{Colors.BOLD}  {score}  {Colors.RESET}")
    _print_nutriscore_bar(score)
    print()

    # Nutriments table
    nutriments = nutrition.get("nutriments", {})
    if nutriments:
        print(f"  {Colors.BOLD}Nutriments (per 100 g):{Colors.RESET}")
        print(f"  {_line('─', 40)}")
        rows = [
            ("Energy",        f"{nutriments.get('energy_kcal', 0):.0f} kcal"),
            ("Fat",           f"{nutriments.get('fat_g', 0):.1f} g"),
            ("  Saturated",   f"{nutriments.get('saturated_fat_g', 0):.1f} g"),
            ("Sugars",        f"{nutriments.get('sugars_g', 0):.1f} g"),
            ("Salt",          f"{nutriments.get('salt_g', 0):.2f} g"),
            ("Proteins",      f"{nutriments.get('proteins_g', 0):.1f} g"),
            ("Fiber",         f"{nutriments.get('fiber_g', 0):.1f} g"),
        ]
        for label, value in rows:
            print(f"  {label:<16} {value:>10}")
        print(f"  {_line('─', 40)}")
    print()

    # Allergens
    allergens = nutrition.get("allergens", "None listed")
    if allergens and allergens != "None listed":
        print(f"  {Colors.RED}{Colors.BOLD}⚠ Allergens:{Colors.RESET} "
              f"{allergens}")
    else:
        print(f"  {Colors.GREEN}✓ No allergens listed{Colors.RESET}")

    # Ingredients (truncated)
    ingredients = nutrition.get("ingredients", "N/A")
    if ingredients and ingredients != "N/A":
        if len(ingredients) > 200:
            ingredients = ingredients[:200] + "..."
        print(f"\n  {Colors.BOLD}Ingredients:{Colors.RESET}")
        print(f"  {Colors.DIM}{ingredients}{Colors.RESET}")

    print()


def _print_nutriscore_bar(score: str) -> None:
    """Print a visual Nutri-Score A–E bar."""
    grades = ["A", "B", "C", "D", "E"]
    bar_parts = []
    for g in grades:
        color = NUTRI_COLORS.get(g, Colors.DIM)
        if g == score:
            bar_parts.append(f"{color}{Colors.BOLD}[{g}]{Colors.RESET}")
        else:
            bar_parts.append(f"{Colors.DIM} {g} {Colors.RESET}")
    print(f"  {''.join(bar_parts)}")


# ── Price comparison table ───────────────────────────────────────────

def print_price_comparison(product_info: dict) -> None:
    """Print a price comparison table across stores.

    Args:
        product_info: Unified product dict from product_mapper.
    """
    prices = product_info.get("prices", [])
    name = product_info.get("name", "Unknown")

    print(_header(f"PRICE COMPARISON — {name}"))

    if not prices:
        print(f"  {Colors.DIM}No price data available for this product."
              f"{Colors.RESET}\n")
        return

    # Table header
    print(f"  {'Store':<20} {'Price':>10}   {'Rating':>8}")
    print(f"  {_line('─', 42)}")

    cheapest = min(prices, key=lambda p: p["price"])["price"]

    for row in prices:
        store = row["store"]
        price = row["price"]
        currency = row.get("currency", "USD")

        # Tag cheapest
        if price == cheapest:
            tag = f"{Colors.GREEN}★ Best{Colors.RESET}"
            price_str = f"{Colors.GREEN}{Colors.BOLD}${price:.2f}{Colors.RESET}"
        else:
            diff = ((price - cheapest) / cheapest) * 100
            tag = f"{Colors.DIM}+{diff:.0f}%{Colors.RESET}"
            price_str = f"${price:.2f}"

        print(f"  {store:<20} {price_str:>20}   {tag:>18}")

    print(f"  {_line('─', 42)}")

    # Savings summary
    if len(prices) > 1:
        most_expensive = max(prices, key=lambda p: p["price"])["price"]
        savings = most_expensive - cheapest
        print(f"\n  {Colors.GREEN}💰 You save up to ${savings:.2f} "
              f"by choosing the cheapest store!{Colors.RESET}")

    print()


# ── Full product display ─────────────────────────────────────────────

def print_product_full(product_info: dict) -> None:
    """Print full information for a product (nutrition + prices)."""
    print_nutrition(product_info)
    print_price_comparison(product_info)


def print_all_products(products: list[dict], mode: str = "full") -> None:
    """Print information for all detected products.

    Args:
        products: List of unified product dicts.
        mode: ``"nutrition"``, ``"price"``, or ``"full"``.
    """
    for i, product in enumerate(products):
        if i > 0:
            print(f"\n{Colors.DIM}{_line('━', 60)}{Colors.RESET}\n")

        if mode == "nutrition":
            print_nutrition(product)
        elif mode == "price":
            print_price_comparison(product)
        else:
            print_product_full(product)


def print_error(message: str) -> None:
    """Print a formatted error message."""
    print(f"\n  {Colors.RED}{Colors.BOLD}✗ Error:{Colors.RESET} {message}\n")


def print_banner() -> None:
    """Print the application launch banner."""
    banner = f"""
{Colors.BOLD}{Colors.CYAN}
  ╔══════════════════════════════════════════════════════╗
  ║       🛒  Personalized Grocery Assistant  🛒        ║
  ║      Smart Vision  •  Nutrition  •  Prices          ║
  ╚══════════════════════════════════════════════════════╝
{Colors.RESET}"""
    print(banner)
