import cv2
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["embedding"]
collection = db["user_embedding"]
user_objects = []
embeddings = []
documents = collection.find()

model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(keep_all=False, device='cpu')

def get_all():
    for document in documents:
        user_id = document['_id']
        user_name = document['userName']
        embedding_list = document['embedding']
        embedding_np = torch.tensor(embedding_list).reshape(1, -1)
        user = {
            'id': user_id,
            'userName': user_name,
        }
        user_objects.append(user)
        embeddings.append(embedding_np)

def extract_video_embeddings(video_path,user_id,  frames_per_second=9):
    get_all()
    video = cv2.VideoCapture(video_path)
    original_fps = video.get(cv2.CAP_PROP_FPS)
    all_embeddings = []
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_interval = int(original_fps / frames_per_second)
        if frame_count % frame_interval == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                face = mtcnn(pil_image)
                if face is not None:
                    face = face.unsqueeze(0)
                    embedding = model(face)
                    all_embeddings.append(embedding)
            except Exception as e:
                print(f"Không thể trích xuất embedding từ frame: {e}")
        frame_count += 1
    video.release()

    if all_embeddings:
        aggregated_embedding = torch.mean(torch.cat(all_embeddings, dim=0), dim=0, keepdim=True)
        update_data(aggregated_embedding, user_id)
        return aggregated_embedding
    else:
        return None

def update_data(embedding, user_id):
    embedding = embedding.tolist()
    collection.update_one({"_id": user_id}, {"$set": {"embedding": embedding}})
    



