# Zepto Data & AI Platform

## Module 1 — Data Pipeline

This project implements Module 1 of the Zepto Data & AI Platform capstone project.

The pipeline follows the required flow:

**Scrape → Clean → Convert → Store → Query**

---

## 1. Project Objective

The objective of Module 1 is to automatically collect book data from BooksToScrape, clean and transform the data, store it in a relational SQLite database, and perform SQL and pandas analysis.

The pipeline collects the following fields:

- Book title
- Price in GBP
- Star rating
- Availability
- Category

The cleaned dataset also contains a converted INR price.

---

## 2. Technologies Used

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- SQL
- NumPy

---

## 3. Data Collection

Data is collected automatically using:

- `requests`
- `BeautifulSoup`

No manual copy/paste is used.

The pipeline collects books from three categories:

1. Travel
2. Mystery
3. Historical Fiction

A total of **61 books** are collected.

This satisfies the requirement of:

- At least 60 books
- At least 3 categories

---

## 4. Data Cleaning

The raw scraped data is transformed into correctly typed fields.

### Price

Example:

```text
£45.17 → 45.17
```
