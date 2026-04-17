from embeddings import create_embeddings
from recognize import recognize_faces
from attendance import save_attendance
from visualize import show_graph

# Step 1: Create embeddings
known_embeddings, known_names = create_embeddings()

# Step 2: Recognize faces
attendance = recognize_faces(known_embeddings, known_names)

# Step 3: Save attendance
save_attendance(attendance)

# Step 4: Show graph
show_graph()