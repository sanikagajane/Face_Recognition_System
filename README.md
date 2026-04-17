# Face Recognition System

This project uses Deep Learning to automatically mark attendance from a group image.

# Features
- Face detection using MTCNN
- Face recognition using FaceNet
- Automatic attendance marking
- Data visualization using Seaborn

# Technologies Used
- Python
- OpenCV
- NumPy
- Pandas
- Seaborn
- Matplotlib

# Project Structure
- embeddings.py → Create face embeddings
- recognize.py → Detect & recognize faces
- attendance.py → Save attendance
- visualize.py → Graph analysis
- main.py → Run full project
- Dataset (Not Included)
  - The dataset is not uploaded due to size and privacy reasons.
  - How to Create Dataset
  - Create a folder named dataset
  - Inside it, create folders for each person:
  - dataset/
    ├── Alice/
    ├── Bob/

  - Add 10–15 face images for each person
  - Use clear, front-facing images with good lighting

# How to Run
```bash
pip install keras-facenet mtcnn opencv-python numpy pandas matplotlib seaborn
python main.py

**#  Output**
- Detects faces in group image
- Recognizes persons using FaceNet
- Saves attendance in CSV
- Displays graph using Seaborn

⭐ If you like this project, give it a star!



