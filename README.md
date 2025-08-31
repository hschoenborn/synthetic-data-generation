# Application: Generative AI for Time Series Data

## Summary

This project focuses on the generation and application of synthetic time series data using advanced generative AI models. The research addresses the challenges posed by limited access to real network usage data due to privacy and security constraints. By leveraging tools such as the **Synthetic Data Vault (SDV)** library, models like **TVAE**, **CTGAN**, and **PARSynthesizer** were implemented and evaluated to produce realistic synthetic datasets that mimic real-world network scenarios.  

These datasets support data-driven analysis while strictly adhering to privacy standards. Additionally, a modular application for data generation and analysis was developed, allowing researchers and business users to seamlessly generate and work with synthetic network data. The results demonstrate the practical value and potential of synthetic data for IT network management and broader analytical applications.

---

## Motivation

In a data-driven era, real data forms the backbone for the development, validation, and continuous improvement of modern software solutions. Many organisations, however, face challenges in accessing production data due to privacy regulations, high costs, or limited data quality. This restricts the ability to simulate, test, and optimise key applications effectively.  

Synthetic data provides a solution by replicating the statistical properties of real datasets without exposing sensitive information. It enables testing scenarios, exploratory analyses, and algorithm training under realistic conditions while ensuring compliance with strict data protection requirements.

---

## Project Focus: Network Usage Data

Network usage data is crucial for IT management, providing insights into network resource performance, identifying bottlenecks, and uncovering potential vulnerabilities. This project focuses on generating synthetic network usage data that closely resembles real-world patterns. Such data allows for comprehensive analysis, testing, and algorithm development without compromising sensitive information or violating regulatory requirements.

The platform is designed for local deployment, ensuring that sensitive data never leaves the organisation. This approach safeguards confidential information while offering a flexible solution tailored to specific operational needs.  

---

## Features

- **Generative AI Models**: Implementation and evaluation of SDV models (TVAE, CTGAN, PARSynthesizer) to produce realistic synthetic time series data.  
- **Modular Application**: Provides a streamlined interface for generating and analysing synthetic network data.  
- **Dockerised Deployment**: The entire application is containerised with Docker, ensuring easy installation, portability, and reproducibility across different environments.  
- **Efficient Training**: Models can be trained with limited computing resources, allowing installation even on machines without dedicated GPUs.  
- **Flexible Use Cases**: Synthetic data can be used for testing, development, research, and other data-driven applications without privacy concerns.  

---

## Technology Stack

- **Python** – Core programming language for model implementation and data processing  
- **SDV (Synthetic Data Vault)** – Library for synthetic data generation  
- **Docker** – Containerisation platform for simplified deployment and environment consistency  
- **Jupyter Notebooks** – For backend scripting and experimentation  
- **Streamlit** – For a lightweight, interactive front-end  

---

## Benefits of Synthetic Data

- Protects sensitive information while replicating realistic data patterns  
- Enables robust testing, analysis, and algorithm development without access to production data  
- Facilitates research by providing large-scale datasets for experimentation  
- Reduces reliance on costly or restricted real-world datasets  
- Provides a reproducible, privacy-compliant solution for time series analysis  

---

## Getting Started

### Prerequisites

- Docker installed on your local machine  
- Basic knowledge of running Docker containers  

### Running the Application

Docker installed on your machine (https://www.docker.com/products/docker-desktop/)

Steps to run the app:

1. Clone the repository
2. Open a terminal and navigate to the root of the project
3. Run the following command: `docker-compose up --build`
