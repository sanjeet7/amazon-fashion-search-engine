# Take Home Assessment Requirements

This document outlines the deliverables for the take-home assessment.

## 1. Project Overview

The goal is to build a semantic recommendation feature for an e-commerce website's fashion product line. This involves creating a simple microservice that can parse a user's natural-language query and find relevant products from the provided dataset using semantic search and Large Language Models (LLMs).

## 2. Deliverables

### 2.1. Architecture Diagram

- A high-level view of the system architecture.
- Format: JPEG or PDF.
- Content: Must show the flow of data from the provided dataset to the user's query response.

### 2.2. Full Executable Code (Microservice)

- The code should be clear and modular.
- It must be executable and demonstrate the functionality out-of-the-box with minimal setup (e.g., adding API keys).
- The microservice should expose its functionality via one of the following:
  - A function
  - A command-line tool
  - An API endpoint

### 2.3. README.md

A comprehensive README file explaining:
- **Project Setup:** Instructions on how to install and run the project.
- **Sample Usage:** Examples of test queries and the resulting recommendations.
- **Design Decisions:** Key design decisions and trade-offs made during development.

### 2.4. (Optional) Additional Exploration

- Any notebooks or scripts used for data exploration, embedding experimentation, or LLM prompt engineering.
- Additional documentation clarifying the approach or next steps.

## 3. Submission

- The entire project should be submitted as a single `.zip` file.
- The submission must be made within five business days.

## 4. Data

The project will use the Amazon Fashion dataset. The key fields are:
- `main_category` (str)
- `title` (str)
- `average_rating` (float)
- `rating_number` (int)
- `features` (list)
- `description` (list)
- `price` (float)
- `images` (list)
- `videos` (list)
- `store` (str)
- `categories` (list)
- `details` (dict)
- `parent_asin` (str)
- `bought_together` (list) 