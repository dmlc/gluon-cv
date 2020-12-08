import os

# os.system("python train_auto_center_net.py --dataset 'chess' --num-trials 25 2>&1 | tee log_center_net_chess_trials_25.txt")
# os.system("python train_auto_center_net.py --dataset 'raccoon' --num-trials 25 2>&1 | tee log_center_net_raccoon_trials_25.txt")
# os.system("python train_auto_center_net.py --dataset 'sheep' --num-trials 25 2>&1 | tee log_center_net_sheep_trials_25.txt")
os.system("python train_auto_center_net.py --dataset 'drone' --num-trials 25 2>&1 | tee log_center_net_drone_trials_25.txt")
os.system("python train_auto_center_net.py --dataset 'car-license-plates' --num-trials 25 2>&1 | tee log_center_net_car_license_plates_trials_25.txt")
os.system("python train_auto_center_net.py --dataset 'hands' --num-trials 25 2>&1 | tee log_center_net_hands_trials_25.txt")
os.system("python train_auto_center_net.py --dataset 'plant-doc' --num-trials 25 2>&1 | tee log_center_net_plant_doc_trials_25.txt")
os.system("python train_auto_center_net.py --dataset 'pets' --num-trials 25 2>&1 | tee log_center_net_pets_trials_25.txt")
os.system("python train_auto_center_net.py --dataset 'cars-and-traffic-signs' --num-trials 25 2>&1 | tee log_center_net_cars_and_traffic_signs_trials_25.txt")
os.system("python train_auto_center_net.py --dataset 'stanford-dogs' --num-trials 25 2>&1 | tee log_center_net_stanford_dogs_trials_25.txt")
