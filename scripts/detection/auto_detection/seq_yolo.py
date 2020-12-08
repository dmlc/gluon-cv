import os

os.system("python train_auto_yolo.py --dataset 'chess' --num-trials 25 2>&1 | tee log_yolo_chess_trials_25.txt")
# os.system("python train_auto_yolo.py --dataset 'raccoon' --num-trials 25 2>&1 | tee log_yolo_raccoon_trials_25.txt")
os.system("python train_auto_yolo.py --dataset 'sheep' --num-trials 25 2>&1 | tee log_yolo_sheep_trials_25.txt")
os.system("python train_auto_yolo.py --dataset 'drone' --num-trials 25 2>&1 | tee log_yolo_drone_trials_25.txt")
os.system("python train_auto_yolo.py --dataset 'car-license-plates' --num-trials 25 2>&1 | tee log_yolo_car_license_plates_trials_25.txt")
os.system("python train_auto_yolo.py --dataset 'hands' --num-trials 25 2>&1 | tee log_yolo_hands_trials_25.txt")
os.system("python train_auto_yolo.py --dataset 'plant-doc' --num-trials 25 2>&1 | tee log_yolo_plant_doc_trials_25.txt")
os.system("python train_auto_yolo.py --dataset 'pets' --num-trials 25 2>&1 | tee log_yolo_pets_trials_25.txt")
os.system("python train_auto_yolo.py --dataset 'cars-and-traffic-signs' --num-trials 25 2>&1 | tee log_yolo_cars_and_traffic_signs_trials_25.txt")
os.system("python train_auto_yolo.py --dataset 'stanford-dogs' --num-trials 25 2>&1 | tee log_yolo_stanford_dogs_trials_25.txt")
