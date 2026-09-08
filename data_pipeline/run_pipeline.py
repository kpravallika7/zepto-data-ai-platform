import sqlite3
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
SQL_OUTPUT_DIR = BASE_DIR / "sql_outputs"

DATA_DIR.mkdir(exist_ok=True)
SQL_OUTPUT_DIR.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "books.db"

BASE_URL = "https://books.toscrape.com/"

CATEGORY_URL = urljoin(
    BASE_URL,
    "catalogue/category/books_1/index.html"
)


# Required conversation rate from the project
GBP_TO_INR = 105.50

# Convert website star ratings into numbers
RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}

# ============================================================
# 2. DOWNLOAD A WEBPAGE
# ============================================================

def get_soup(url):
    """
    Download a webpage and return its HTML as a BeautifulSoup object.
    """

    response = requests.get(
        url,
        timeout=30,
        headers={
            "user-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")

# ============================================================
# 3. SCRAPE CATEGORIES
# ============================================================

def get_categories():
    """
    Get book categories from BooksToScrape.
    """

    soup = get_soup(CATEGORY_URL)

    categories = []

    for link in soup.select("ul.nav-list ul li a"):

        category_name = link.get_text(strip=True)
        category_href = link.get("href")

        if category_name and category_href:

            category_url = urljoin(
                CATEGORY_URL,
                category_href
            )

            categories.append({
                "category_name": category_name,
                "url": category_url
            })

    return categories


# ============================================================
# 4. EXTRACT STAR RATING
# ============================================================

def extract_rating(article):
    """
    Convert the website's star rating into a number from 1 to 5.
    """

    rating_tag = article.select_one("p.star-rating")

    if not rating_tag:
        return None

    classes = rating_tag.get("class", [])

    for class_name in classes:

        if class_name in RATING_MAP:

            return RATING_MAP[class_name]

    return None


# ============================================================
# 5. SCRAPE BOOKS FROM ONE CATEGORY
# ============================================================

def scrape_category(category_name, category_url, max_books=25):
    """
    Scrape books from one category.

    The function follows the category's pagination until
    enough books have been collected.
    """

    books = []
    current_url = category_url

    while current_url:

        print(f"  Opening: {current_url}")

        soup = get_soup(current_url)

        book_articles = soup.select("article.product_pod")

        for article in book_articles:

            title_tag = article.select_one("h3 a")
            price_tag = article.select_one(".price_color")
            availability_tag = article.select_one(".availability")

            if not title_tag:
                continue

            title = title_tag.get("title")

            if not title:
                title = title_tag.get_text(strip=True)

            price = None

            if price_tag:
                price = price_tag.get_text(" ", strip=True)
                price = price.replace("Â£", "£").strip()

            availability = None

            if availability_tag:
                availability = availability_tag.get_text(
                    " ",
                    strip=True
                )

            rating = extract_rating(article)

            books.append({
                "title": title,
                "price": price,
                "rating": rating,
                "availability": availability,
                "category": category_name
            })

            if len(books) >= max_books:
                return books

        next_link = soup.select_one("li.next a")

        if next_link:

            current_url = urljoin(
                current_url,
                next_link.get("href")
            )

        else:

            current_url = None

    return books


# ============================================================
# 6. SCRAPE AT LEAST 60 BOOKS FROM AT LEAST 3 CATEGORIES
# ============================================================

def scrape_books():
    """
    Collect books from the first three available categories.

    We collect up to 25 books per category.
    Therefore, the target is up to 75 books.
    """

    categories = get_categories()

    if len(categories) < 3:
        raise RuntimeError(
            "BooksToScrape returned fewer than 3 categories."
        )

    selected_categories = categories[:3]

    print("\nSelected categories:")

    for category in selected_categories:
        print(f"- {category['category_name']}")

    all_books = []

    for category in selected_categories:

        print(
            f"\nScraping category: "
            f"{category['category_name']}"
        )

        category_books = scrape_category(
            category["category_name"],
            category["url"],
            max_books=25
        )

        print(
            f"Books collected from "
            f"{category['category_name']}: "
            f"{len(category_books)}"
        )

        all_books.extend(category_books)

    return all_books


# ============================================================
# 7. CLEAN THE DATA
# ============================================================

def clean_data(records):
    """
    Convert scraped data into clean, correctly typed data.
    """

    df = pd.DataFrame(records)

    # Remove duplicate books within the same category
    df = df.drop_duplicates(
        subset=["title", "category"]
    ).copy()

    # Fix common UTF-8/Windows-1252 encoding artifacts
    df["title"] = (
        df["title"]
        .str.replace("â€™", "’", regex=False)
        .str.replace("â€œ", "“", regex=False)
        .str.replace("â€\x9d", "”", regex=False)
        .str.replace("â€“", "–", regex=False)
        .str.replace("â€”", "—", regex=False)
        .str.replace("â\x80\x99", "’", regex=False)
    )

    # --------------------------------------------------------
    # PRICE
    # Example:
    # "£25.00" -> 25.00
    # --------------------------------------------------------

    df["price_gbp"] = (
        df["price"]
        .astype(str)
        .str.replace("£", "", regex=False)
        .str.replace("Â", "", regex=False)
        .str.strip()
    )

    df["price_gbp"] = pd.to_numeric(
        df["price_gbp"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # RATING
    # Example:
    # One -> 1
    # Five -> 5
    # --------------------------------------------------------

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    # Convert rating to nullable integer
    df["rating"] = df["rating"].astype("Int64")

    # --------------------------------------------------------
    # AVAILABILITY
    # Example:
    # "In stock (22 available)" -> True
    # --------------------------------------------------------

    df["in_stock"] = (
        df["availability"]
        .astype(str)
        .str.contains(
            "In stock",
            case=False,
            na=False
        )
    )

    # --------------------------------------------------------
    # GBP TO INR CONVERSION
    # Required rate:
    # 1 GBP = 105.50 INR
    # --------------------------------------------------------

    df["price_inr"] = (
        df["price_gbp"] * GBP_TO_INR
    ).round(2)

    # Keep the clean columns needed by the project
    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    return df


# ============================================================
# 8. CREATE SQLITE DATABASE
# ============================================================

def create_database(df):
    """
    Create two related tables:

    categories
    books

    books.category_id is a foreign key
    referencing categories.category_id.
    """

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Remove old tables if they already exist
    # --------------------------------------------------------

    cursor.execute(
        "DROP TABLE IF EXISTS books"
    )

    cursor.execute(
        "DROP TABLE IF EXISTS categories"
    )

    # --------------------------------------------------------
    # CATEGORIES TABLE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
        """
    )

    # --------------------------------------------------------
    # BOOKS TABLE
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER,

            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    # --------------------------------------------------------
    # INSERT CATEGORIES
    # --------------------------------------------------------

    categories = (
        df["category"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    for category in categories:

        cursor.execute(
            """
            INSERT INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    # Create category name -> category ID mapping
    category_map = dict(
        cursor.execute(
            """
            SELECT category_name, category_id
            FROM categories
            """
        ).fetchall()
    )

    # --------------------------------------------------------
    # INSERT BOOKS
    # --------------------------------------------------------

    for _, row in df.iterrows():

        rating = None

        if pd.notna(row["rating"]):
            rating = int(row["rating"])

        cursor.execute(
            """
            INSERT INTO books (
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                rating,
                int(row["in_stock"]),
                category_map[row["category"]]
            )
        )

    connection.commit()

    print(
        f"\nSQLite database created at: {DB_PATH}"
    )

    return connection


# ============================================================
# 9. RUN REQUIRED SQL QUERIES
# ============================================================

def run_sql_queries(connection):
    """
    Run the SQL queries required by Module 1.

    Required query types demonstrated:

    1. SELECT + WHERE
    2. ORDER BY + LIMIT
    3. DISTINCT
    4. BETWEEN
    5. IN
    6. JOIN
    """

    # --------------------------------------------------------
    # QUERY 1
    # SELECT + WHERE
    # --------------------------------------------------------

    query1 = """
        SELECT
            title,
            price_inr
        FROM books
        WHERE price_inr > 5000
    """

    result1 = pd.read_sql_query(
        query1,
        connection
    )

    result1.to_csv(
        SQL_OUTPUT_DIR / "01_select_where.csv",
        index=False
    )

    # --------------------------------------------------------
    # QUERY 2
    # ORDER BY + LIMIT
    # --------------------------------------------------------

    query2 = """
        SELECT
            title,
            price_inr
        FROM books
        ORDER BY price_inr DESC
        LIMIT 10
    """

    result2 = pd.read_sql_query(
        query2,
        connection
    )

    result2.to_csv(
        SQL_OUTPUT_DIR / "02_order_by_limit.csv",
        index=False
    )

    # --------------------------------------------------------
    # QUERY 3
    # DISTINCT
    # --------------------------------------------------------

    query3 = """
        SELECT DISTINCT
            rating
        FROM books
        ORDER BY rating
    """

    result3 = pd.read_sql_query(
        query3,
        connection
    )

    result3.to_csv(
        SQL_OUTPUT_DIR / "03_distinct.csv",
        index=False
    )

    # --------------------------------------------------------
    # QUERY 4
    # BETWEEN
    # --------------------------------------------------------

    query4 = """
        SELECT
            title,
            price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
    """

    result4 = pd.read_sql_query(
        query4,
        connection
    )

    result4.to_csv(
        SQL_OUTPUT_DIR / "04_between.csv",
        index=False
    )

    # --------------------------------------------------------
    # QUERY 5
    # IN
    # --------------------------------------------------------

    query5 = """
        SELECT
            title,
            rating
        FROM books
        WHERE rating IN (4, 5)
    """

    result5 = pd.read_sql_query(
        query5,
        connection
    )

    result5.to_csv(
        SQL_OUTPUT_DIR / "05_in.csv",
        index=False
    )

    # --------------------------------------------------------
    # QUERY 6
    # JOIN
    # --------------------------------------------------------

    query6 = """
        SELECT
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY
            c.category_name,
            b.title
    """

    result6 = pd.read_sql_query(
        query6,
        connection
    )

    result6.to_csv(
        SQL_OUTPUT_DIR / "06_join.csv",
        index=False
    )

    print("\nSQL queries completed.")

    return result6


# ============================================================
# 10. VERIFY JOIN USING PANDAS MERGE
# ============================================================

def verify_with_pandas(sql_join_result):
    """
    Reproduce the SQL JOIN using pandas.merge()
    and compare the result.
    """

    connection = sqlite3.connect(DB_PATH)

    # Read books table into pandas
    books_df = pd.read_sql_query(
        """
        SELECT
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        connection
    )

    # Read categories table into pandas
    categories_df = pd.read_sql_query(
        """
        SELECT
            category_id,
            category_name
        FROM categories
        """,
        connection
    )

    connection.close()

    # --------------------------------------------------------
    # REPRODUCE THE JOIN USING pandas.merge()
    # --------------------------------------------------------

    pandas_join = books_df.merge(
        categories_df,
        on="category_id",
        how="inner"
    )

    pandas_join = pandas_join[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name"
        ]
    ]

    # --------------------------------------------------------
    # SORT BOTH RESULTS BEFORE COMPARISON
    # --------------------------------------------------------

    sql_sorted = (
        sql_join_result
        .sort_values(
            ["category_name", "title"]
        )
        .reset_index(drop=True)
    )

    pandas_sorted = (
        pandas_join
        .sort_values(
            ["category_name", "title"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # CHECK WHETHER BOTH RESULTS ARE IDENTICAL
    # --------------------------------------------------------

    equivalent = sql_sorted.equals(
        pandas_sorted
    )

    print("\nPandas JOIN verification:")
    print(
        f"SQL JOIN and pandas.merge() equivalent: "
        f"{equivalent}"
    )

    # Save pandas verification output
    pandas_sorted.to_csv(
        SQL_OUTPUT_DIR / "pandas_merge_verification.csv",
        index=False
    )

    return equivalent


# ============================================================
# 11. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("MODULE 1 - DATA PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # SCRAPE
    # --------------------------------------------------------

    print("\nSTEP 1: SCRAPING")

    records = scrape_books()

    print(
        f"\nTotal raw books collected: {len(records)}"
    )

    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    print("\nSTEP 2: CLEANING")

    df = clean_data(records)

    print(
        f"Cleaned books: {len(df)}"
    )

    print(
        f"Categories represented: "
        f"{df['category'].nunique()}"
    )

    # --------------------------------------------------------
    # CHECK PROJECT REQUIREMENTS
    # --------------------------------------------------------

    if len(df) < 60:

        raise RuntimeError(
            "ERROR: Fewer than 60 books were collected."
        )

    if df["category"].nunique() < 3:

        raise RuntimeError(
            "ERROR: Fewer than 3 categories were collected."
        )

    # --------------------------------------------------------
    # CONVERT / SAVE CLEAN DATA
    # --------------------------------------------------------

    print("\nSTEP 3: CONVERTING")

    clean_csv = DATA_DIR / "books_clean.csv"

    df.to_csv(
        clean_csv,
        index=False
    )

    print(
        f"Clean data saved to: {clean_csv}"
    )

    print("\nSample clean data:")

    print(
        df.head().to_string(index=False)
    )

    # --------------------------------------------------------
    # STORE IN SQLITE
    # --------------------------------------------------------

    print("\nSTEP 4: STORING")

    connection = create_database(df)

    # --------------------------------------------------------
    # SQL QUERIES
    # --------------------------------------------------------

    print("\nSTEP 5: QUERYING")

    sql_join_result = run_sql_queries(
        connection
    )

    connection.close()

    # --------------------------------------------------------
    # PANDAS VERIFICATION
    # --------------------------------------------------------

    print("\nSTEP 6: PANDAS VERIFICATION")

    join_verified = verify_with_pandas(
        sql_join_result
    )

    if not join_verified:

        raise RuntimeError(
            "ERROR: SQL JOIN and pandas.merge() "
            "results do not match."
        )

    # --------------------------------------------------------
    # FINAL SUCCESS MESSAGE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODULE 1 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Books collected: {len(df)}")
    print(f"Categories: {df['category'].nunique()}")
    print("SQLite database: CREATED")
    print("SQL queries: COMPLETED")
    print("Pandas merge verification: PASSED")
    print("=" * 60)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()

