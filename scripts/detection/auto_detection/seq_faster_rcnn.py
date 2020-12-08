import os

os.system("python train_auto_faster_rcnn.py --dataset 'chess' --num-trials 25 2>&1 | tee log_faster_rcnn_chess_trials_25.txt")
os.system("python train_auto_faster_rcnn.py --dataset 'raccoon' --num-trials 25 2>&1 | tee log_faster_rcnn_raccoon_trials_25.txt")
os.system("python train_auto_faster_rcnn.py --dataset 'sheep' --num-trials 25 2>&1 | tee log_faster_rcnn_sheep_trials_25.txt")
os.system("python train_auto_faster_rcnn.py --dataset 'drone' --num-trials 25 2>&1 | tee log_faster_rcnn_drone_trials_25.txt")
os.system("python train_auto_faster_rcnn.py --dataset 'car-license-plates' --num-trials 25 2>&1 | tee log_faster_rcnn_car_license_plates_trials_25.txt")
os.system("python train_auto_faster_rcnn.py --dataset 'hands' --num-trials 25 2>&1 | tee log_faster_rcnn_hands_trials_25.txt")
os.system("python train_auto_faster_rcnn.py --dataset 'plant-doc' --num-trials 25 2>&1 | tee log_faster_rcnn_plant_doc_trials_25.txt")
os.system("python train_auto_faster_rcnn.py --dataset 'pets' --num-trials 25 2>&1 | tee log_faster_rcnn_pets_trials_25.txt")
os.system("python train_auto_faster_rcnn.py --dataset 'cars-and-traffic-signs' --num-trials 25 2>&1 | tee log_faster_rcnn_cars_and_traffic_signs_trials_25.txt")
os.system("python train_auto_faster_rcnn.py --dataset 'stanford-dogs' --num-trials 25 2>&1 | tee log_faster_rcnn_stanford_dogs_trials_25.txt")
