"""A small fictional bookshop database the agent can query. Created on first start."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    city TEXT NOT NULL,
    joined_on TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE order_items (
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
"""

CUSTOMERS = [
    (1, "Asha Verma", "Pune", "2025-01-14"),
    (2, "Rohan Mehta", "Mumbai", "2025-02-03"),
    (3, "Neha Iyer", "Bengaluru", "2025-02-21"),
    (4, "Karan Singh", "Delhi", "2025-03-09"),
    (5, "Priya Nair", "Kochi", "2025-04-17"),
    (6, "Vikram Rao", "Hyderabad", "2025-05-02"),
    (7, "Meera Joshi", "Pune", "2025-06-11"),
    (8, "Arjun Das", "Kolkata", "2025-07-25"),
]

PRODUCTS = [
    (1, "The Quiet Algorithm", "Technology", 499.0),
    (2, "Vectors and Meaning", "Technology", 650.0),
    (3, "Monsoon Letters", "Fiction", 299.0),
    (4, "The Spice Route Atlas", "Travel", 899.0),
    (5, "Small Habits, Big Years", "Self-help", 349.0),
    (6, "Graphs for Everyone", "Technology", 550.0),
    (7, "Lanterns of Varanasi", "Fiction", 399.0),
    (8, "Trekking the Western Ghats", "Travel", 749.0),
    (9, "The Calm Ledger", "Finance", 450.0),
    (10, "Kitchen Chemistry", "Cooking", 520.0),
]

ORDERS = [
    (1, 1, "2025-08-01", "delivered"),
    (2, 2, "2025-08-03", "delivered"),
    (3, 3, "2025-08-07", "delivered"),
    (4, 1, "2025-08-15", "delivered"),
    (5, 4, "2025-08-19", "cancelled"),
    (6, 5, "2025-09-02", "delivered"),
    (7, 6, "2025-09-05", "delivered"),
    (8, 2, "2025-09-12", "delivered"),
    (9, 7, "2025-09-18", "shipped"),
    (10, 8, "2025-09-22", "delivered"),
    (11, 3, "2025-10-01", "delivered"),
    (12, 4, "2025-10-04", "delivered"),
    (13, 5, "2025-10-10", "shipped"),
    (14, 1, "2025-10-14", "delivered"),
    (15, 6, "2025-10-20", "processing"),
    (16, 7, "2025-10-28", "delivered"),
]

ORDER_ITEMS = [
    (1, 1, 1, 499.0), (1, 3, 2, 299.0),
    (2, 2, 1, 650.0),
    (3, 4, 1, 899.0), (3, 5, 1, 349.0),
    (4, 6, 2, 550.0),
    (5, 9, 1, 450.0),
    (6, 7, 1, 399.0), (6, 3, 1, 299.0),
    (7, 2, 1, 650.0), (7, 1, 1, 499.0),
    (8, 8, 1, 749.0),
    (9, 10, 1, 520.0),
    (10, 5, 3, 349.0),
    (11, 6, 1, 550.0), (11, 2, 1, 650.0),
    (12, 4, 1, 899.0),
    (13, 9, 2, 450.0),
    (14, 1, 1, 499.0), (14, 7, 1, 399.0),
    (15, 8, 1, 749.0),
    (16, 3, 1, 299.0), (16, 10, 1, 520.0),
]  # fmt: skip


def ensure_sample_db(path: Path) -> Path:
    """Create the fictional bookshop database if it does not exist yet."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(tmp) as conn:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", PRODUCTS)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", ORDERS)
        conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", ORDER_ITEMS)
    conn.close()
    tmp.replace(path)
    return path
