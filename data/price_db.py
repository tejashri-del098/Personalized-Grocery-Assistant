"""
price_db.py — SQLite Price Comparison Database

Manages a local SQLite database that stores product prices across
multiple stores for the "Price Comparison" feature.

The database auto-seeds with ~15 common grocery products across
3 fictional stores (FreshMart, GreenGrocer, MegaMart) on first run.
"""

import sqlite3
import os

# ── Database path ────────────────────────────────────────────────────
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db")
DB_PATH = os.path.join(DB_DIR, "prices.db")


# ── Seed data ────────────────────────────────────────────────────────
SEED_PRODUCTS = [
    # (product_name, category)
    ("Banana", "Fruit"),
    ("Apple", "Fruit"),
    ("Orange", "Fruit"),
    ("Broccoli", "Vegetable"),
    ("Carrot", "Vegetable"),
    ("Tomato", "Vegetable"),
    ("Milk (1L)", "Dairy"),
    ("Cheddar Cheese (200g)", "Dairy"),
    ("Yogurt (500g)", "Dairy"),
    ("White Bread Loaf", "Bakery"),
    ("Pasta (500g)", "Pantry"),
    ("Rice (1kg)", "Pantry"),
    ("Olive Oil (500ml)", "Pantry"),
    ("Chicken Breast (500g)", "Meat"),
    ("Eggs (12-pack)", "Dairy"),
]

SEED_PRICES = {
    # product_name: (FreshMart, GreenGrocer, MegaMart)
    "Banana":                (1.29, 1.19, 1.39),
    "Apple":                 (2.49, 2.29, 2.59),
    "Orange":                (1.99, 1.89, 2.09),
    "Broccoli":              (2.79, 2.49, 2.99),
    "Carrot":                (1.49, 1.39, 1.59),
    "Tomato":                (3.29, 2.99, 3.49),
    "Milk (1L)":             (1.89, 1.79, 1.99),
    "Cheddar Cheese (200g)": (4.49, 4.29, 4.69),
    "Yogurt (500g)":         (2.99, 2.79, 3.19),
    "White Bread Loaf":      (2.49, 2.39, 2.69),
    "Pasta (500g)":          (1.79, 1.69, 1.89),
    "Rice (1kg)":            (3.49, 3.29, 3.69),
    "Olive Oil (500ml)":     (5.99, 5.49, 6.29),
    "Chicken Breast (500g)": (6.49, 5.99, 6.99),
    "Eggs (12-pack)":        (3.99, 3.79, 4.19),
}

STORES = ["FreshMart", "GreenGrocer", "MegaMart"]


# ── Database initialisation ─────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite price database."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables and seed data if they don't already exist.

    This function is **idempotent** — calling it multiple times will
    not duplicate data.
    """
    conn = _get_connection()
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            category    TEXT    NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stores (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT    NOT NULL UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id  INTEGER NOT NULL,
            store_id    INTEGER NOT NULL,
            price       REAL    NOT NULL,
            currency    TEXT    DEFAULT 'USD',
            FOREIGN KEY (product_id) REFERENCES products(id),
            FOREIGN KEY (store_id)   REFERENCES stores(id),
            UNIQUE(product_id, store_id)
        )
    """)

    # Seed stores
    for store in STORES:
        cursor.execute(
            "INSERT OR IGNORE INTO stores (name) VALUES (?)", (store,)
        )

    # Seed products and prices
    for product_name, category in SEED_PRODUCTS:
        cursor.execute(
            "INSERT OR IGNORE INTO products (name, category) VALUES (?, ?)",
            (product_name, category),
        )
        product_id = cursor.execute(
            "SELECT id FROM products WHERE name = ?", (product_name,)
        ).fetchone()["id"]

        if product_name in SEED_PRICES:
            for idx, store in enumerate(STORES):
                store_id = cursor.execute(
                    "SELECT id FROM stores WHERE name = ?", (store,)
                ).fetchone()["id"]
                price = SEED_PRICES[product_name][idx]
                cursor.execute(
                    "INSERT OR IGNORE INTO prices (product_id, store_id, price) "
                    "VALUES (?, ?, ?)",
                    (product_id, store_id, price),
                )

    conn.commit()
    conn.close()


# ── Query functions ──────────────────────────────────────────────────

def get_price_comparison(product_name: str) -> list[dict]:
    """Get prices for a product across all stores.

    Uses a **case-insensitive fuzzy match** (SQL ``LIKE``) so that
    ``"banana"`` matches ``"Banana"``.

    Args:
        product_name: Name or partial name of the product.

    Returns:
        List of dicts with ``store``, ``price``, ``currency``, and the
        matched ``product_name``.  Sorted by price (cheapest first).
    """
    init_db()  # ensure DB is ready
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT p.name AS product_name,
               s.name AS store,
               pr.price,
               pr.currency
        FROM prices pr
        JOIN products p ON pr.product_id = p.id
        JOIN stores  s ON pr.store_id   = s.id
        WHERE LOWER(p.name) LIKE LOWER(?)
        ORDER BY pr.price ASC
    """, (f"%{product_name}%",))

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "product_name": row["product_name"],
            "store": row["store"],
            "price": row["price"],
            "currency": row["currency"],
        }
        for row in rows
    ]


def get_all_products() -> list[dict]:
    """Return all products in the database.

    Returns:
        List of dicts with ``id``, ``name``, and ``category``.
    """
    init_db()
    conn = _get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, category FROM products ORDER BY name")
    rows = cursor.fetchall()
    conn.close()

    return [{"id": row["id"], "name": row["name"], "category": row["category"]}
            for row in rows]


def get_cheapest_store(product_name: str) -> dict | None:
    """Find the store with the lowest price for a given product.

    Args:
        product_name: Product name (partial match supported).

    Returns:
        Dict with ``store``, ``price``, ``product_name``, or ``None``.
    """
    prices = get_price_comparison(product_name)
    return prices[0] if prices else None
