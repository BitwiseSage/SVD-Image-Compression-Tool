![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-Linear%20Algebra-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

# SVD Image Compression Tool

An interactive image compression web app built using **Singular Value Decomposition (SVD)** for low-rank matrix approximation.  
This project demonstrates how linear algebra techniques reduce storage while preserving image quality.

---

## 📌 Features

- Upload any RGB image
- Adjust compression rank using slider
- Real-time compressed image preview
- Channel-wise RGB SVD compression
- Compression ratio calculation
- Reconstruction error (Frobenius norm)
- Rank vs compression analysis graphs
- Download compressed image output

---

## 🧠 Mathematical Background

Singular Value Decomposition factorizes an image matrix:

A = UΣVᵀ

Instead of storing the full matrix, we approximate it using top **k singular values**:

A_k = U_k Σ_k V_kᵀ

Smaller k → higher compression  
Larger k → better image quality

This creates a trade-off between **storage efficiency** and **reconstruction accuracy**.

---

## 📊 Compression Metrics Used

The application computes:

- Compression Ratio (% storage saved)
- Reconstruction Error (Frobenius Norm)

Frobenius norm:

||A − Aₖ||

Lower error indicates better reconstruction quality.

---

## 🛠 Tech Stack

- Python
- NumPy
- Streamlit
- Pillow
- Matplotlib

---

## 📂 Project Structure
SVD-Image-Compression-Tool/
│
├── app.py
├── svd_utils.py
├── requirements.txt
├── sample_images/
└── README.md

---

## ▶️ How to Run Locally

Clone repository:

git clone https://github.com/BitwiseSage/SVD-Image-Compression-Tool.git

Navigate into folder:


cd SVD-Image-Compression-Tool


Install dependencies:


pip install -r requirements.txt


Run application:


python -m streamlit run app.py

---

## 📈 Applications of SVD Compression

- Image storage optimization
- Facial recognition preprocessing
- Recommender systems
- Signal processing
- Dimensionality reduction (PCA)

---

## 📷 Demo

Example compression output:

pdf file of demo placed in sample_images folder as "demo.pdf"


## 👨‍💻 Authors

Developed as part of a Linear Algebra course project demonstrating practical applications of Singular Value Decomposition (SVD) in image compression and matrix approximation.