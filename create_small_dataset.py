import os
import numpy as np
import cv2

base_path = r"C:\Users\abc\Desktop\DR_Project\Detecting-diabetic-retinopathy\datasets"

classes = ['0', '1', '2', '3', '4']

for folder in ['Train', 'Validation', 'Test']:
    for cls in classes:
        path = os.path.join(base_path, folder, cls)
        os.makedirs(path, exist_ok=True)

        # create 50 images per class for Train
        if folder == 'Train':
            count = 50
        elif folder == 'Validation':
            count = 10
        else:
            count = 10

        for i in range(count):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(path, f"{cls}_{i}.jpg"), img)

print("✅ Small dataset created successfully!")