from ultralytics import YOLO
 
model = YOLO("yolo11n.pt") 
trained = model.train( data = "../data_processing/bdd_to_yolo/dataset_yolo_converted/bdd100k_ultralytics.yaml",
                       epochs = 60,
                       imgsz = 640, 
                       device ="0"
                       ) 
metrics = model.val() 

