from utils.helper import generate_output
from utils.fid import calculate_fid_given_paths
import face_recognition
import os


def validation(config):
    noise_dim = config['noise_dim']
    G = config['generator']
    device = config['device']
    output_path = config['output_path']
    val_path = config['val_path']

    generate_output(noise_dim, 1000, G, device, output_path)

    face_score = face_recog(output_path)
    fid_score = calculate_fid_given_paths([val_path, output_path], batch_size = 50, 
                                         device = device, dims = 2048)

    print('Validation set: Face acc: {:.2f}%, FID = {:.2f}\n'.format(
        face_score, fid_score))
    
    return face_score, fid_score


def face_recog(image_dir):
    image_ids = os.listdir(image_dir)
    total_faces = len(image_ids)
    num_faces = 0
    
    for image_id in image_ids:
        image_path = os.path.join(image_dir, image_id)
        try: # Prevent unexpected file
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image, model="HOG")
            if len(face_locations) == 1:
                num_faces += 1
        except:
            total_faces -= 1
    acc = (num_faces / total_faces) * 100
    return acc