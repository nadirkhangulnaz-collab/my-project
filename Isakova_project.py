# yolo_pose_falldetect_v10.py - Система контроля безопасности (CONFIDENCE УДАЛЕНО С ДИСПЛЕЯ)
import cv2
import numpy as np
import time
import os 
from collections import deque
import requests
from ultralytics import YOLO
BOT_TOKEN = "8498865587:AAGi7sRIXS-cD3zRq-ggj8E74KshU7Aj8Dw" 
CHAT_ID = "1006339136" 
# =====================================================================

# ========== Жүйе Параметрлері ==========
# ========== OpenCV камера ашу (Оптимизировано для Logitech) ==========
# Попробуйте изменить индекс здесь: 0 - встроенная, 1 или 2 - Logitech
VIDEO_SOURCE = 1 

cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print(f"Камера с индексом {VIDEO_SOURCE} не найдена. Пробую индекс 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    
    if not cap.isOpened():
        print("Попытка поиска индекса 2...")
        cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ ҚАТЕ: Камера қосылмады немесе қолжетімсіз.")
    exit(1)

# Настройка разрешения (Logitech StreamCam поддерживает высокое качество, 
# но для YOLO мы оставляем IMG_SIZE для скорости)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)
# Полезно для Logitech: принудительная установка FPS, чтобы избежать тормозов
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS) 

print(f"Жүйе іске қосылды. Камера: {VIDEO_SOURCE}. Шығу үшін ESC басыңыз.")
# ========== Құлауды Анықтау Шегі (Fall detection thresholds) ==========
HIP_DROP_THRESHOLD = 0.03      
NOSE_LOW_THRESHOLD = 0.50      
ANGLE_CHANGE_THRESHOLD = 10.0  
MIN_CONSECUTIVE = 2            
COOLDOWN_SECONDS = 5  
FALL_STAY_THRESHOLD = 10   

# ========== Адам Жоқтығын Анықтау ==========
NO_PERSON_THRESHOLD_SECONDS = 10 
ALERT_COOLDOWN_LONG = 60          

# ********** СЕНІМДІЛІК СТАБИЛЬДІЛІГІН ТЕКСЕРУ **********
CONFIDENCE_TOLERANCE = 0.07      
# *******************************************************

# ========== Модель Жүктеу ==========
print("Модель жүктелуде... Күте тұрыңыз...")
if not os.path.exists("yolov8s-pose.pt"):
    print("❌ ҚАТЕ: Файл 'yolov8s-pose.pt' табылмады! Жүктеңіз оны!")
    exit(1)

model = YOLO("yolov8s-pose.pt") 

# ========== Буферлер ==========
frame_buffer = deque(maxlen=int(BUFFER_SECONDS * TARGET_FPS))
hip_history = deque(maxlen=TARGET_FPS)
angle_history = deque(maxlen=TARGET_FPS)
nose_history = deque(maxlen=TARGET_FPS)

# ========== Уақыт Айнымалылары ==========
fall_counter = 0
last_fall_alert_time = 0

last_initial_fall_time = 0      # Время первого обнаружения падения
is_initial_alert_sent = False
is_critical_alert_sent = False

last_person_detection_time = time.time() 
last_long_alert_time = 0               

# Сенімділік (Confidence) үшін 
last_stable_confidence_time = time.time()
stable_confidence_base = 0.0 

# Адамның жоқтығын тексеру үшін
is_no_person_alert_sent = False 

# =====================================================================
# ========== 2. КӨМЕКШІ ФУНКЦИЯЛАР ==========

def send_telegram_alert(image_path, caption):
    """Отправка сообщения в Telegram."""
    if not os.path.exists(image_path):
        print(f"[Telegram ERROR] Файл табылмады: {image_path}")
        return
        
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with open(image_path, 'rb') as f:
            files = {'photo': f}
            data = {'chat_id': CHAT_ID, 'caption': caption}
            print("Telegram-ға жіберілуде...")
            response = requests.post(url, data=data, files=files, timeout=10)
            
        if response.status_code == 200:
            print(f"[Telegram] Хабарлама СӘТТІ жіберілді!")
        else:
            print(f"[Telegram ERROR] Қате код: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"[Telegram ERROR] Интернет қатесі: {e}")
    except Exception as e:
        print(f"[Telegram ERROR] Басқа қате: {e}")

def compute_body_angle(landmarks, img_h, img_w):
    """Вычисляет угол корпуса."""
    try:
        left_sh = landmarks[5]
        right_sh = landmarks[6]
        left_hip = landmarks[11]
        right_hip = landmarks[12]
        
        if left_hip[2] < 0.1 or right_hip[2] < 0.1:
            return None
            
    except Exception:
        return None

    sx = ((left_sh[0] + right_sh[0]) / 2.0) 
    sy = ((left_sh[1] + right_sh[1]) / 2.0)
    hx = ((left_hip[0] + right_hip[0]) / 2.0) 
    hy = ((left_hip[1] + right_hip[1]) / 2.0) 

    dx = hx - sx
    dy = hy - sy
    ang = np.degrees(np.arctan2(dy, dx))
    rel = abs(90 - abs(ang)) 
    return rel

# =====================================================================
# ========== 3. НЕГІЗГІ ОРЫНДАУ БЛОГЫ ==========

# ========== OpenCV камера ашу ==========
cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_DSHOW) 

if not cap.isOpened():
    print("Негізгі камера ашылмады. 1-камераны тексеру...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
    
    if not cap.isOpened():
        print("❌ ҚАТЕ: Камера қосылмады немесе қолжетімсіз.")
        exit(1)

# set desired size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)

print("Жүйе іске қосылды. Шығу үшін ESC басыңыз.")

while True:
    now = time.time()
    start_time = now
    ret, frame = cap.read() 
    if not ret:
        print("Кадр оқылмады. Қайта қосылу...")
        time.sleep(0.1) 
        continue 
    
    h, w = frame.shape[:2]

    # 1. Модельді іске қосу
    results = model(frame, imgsz=IMG_SIZE, device='cpu', conf=0.15, verbose=False)
    
    # 2. Скелетті суретке салу
    annotated_frame = results[0].plot() 

    # Статус айнымалылары
    fall_detected = False
    no_person_detected = False
    
    status_text = "Status: OK"
    status_color = (0, 255, 0) # Жасыл

    # 3. Деректерді алу
    res = results[0]
    landmarks = None
    main_conf_score = 0.0 # Енді display-де көрсетілмейді

    is_person_visible = False
    if hasattr(res, "keypoints") and res.keypoints is not None and res.keypoints.xy.shape[0] > 0:
        is_person_visible = True
        last_person_detection_time = now

        # ********** ЛОГИКА: Адам оралды **********
        if is_no_person_alert_sent:
            is_no_person_alert_sent = False
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            img_name = f"person_returned_{timestamp}.jpg"
            cv2.imwrite(img_name, frame)
            
            print("✅ АДАМ ҚАЙТА ОРАЛДЫ!")
            msg = f"✅ АДАМ ҚАЙТА ОРАЛДЫ!\nУақыты: {time.strftime('%H:%M:%S')}"
            send_telegram_alert(img_name, msg)
        # **********************************************

        # Анықталған адамның негізгі сенімділік мәнін алу 
        main_conf_score = res.boxes.conf[0].item() if len(res.boxes.conf) > 0 else 0.0
        
        landmarks_xy = res.keypoints.xy[0].cpu().numpy()
        landmarks_conf = res.keypoints.conf[0].cpu().numpy().reshape(-1, 1)
        landmarks = np.hstack((landmarks_xy, landmarks_conf))
        
        if len(landmarks) >= 13: 
            try:
                nose_y_norm = landmarks[0][1] / h 
                left_hip_y = landmarks[11][1]
                right_hip_y = landmarks[12][1]
                hip_conf = (landmarks[11][2] + landmarks[12][2]) / 2
            except IndexError:
                nose_y_norm = 0
                hip_conf = 0

            # --- А. Сенімділікті тексеру ---
            if stable_confidence_base == 0.0:
                stable_confidence_base = main_conf_score 
                last_stable_confidence_time = now
                
            conf_delta = abs(main_conf_score - stable_confidence_base)
            
            if conf_delta > CONFIDENCE_TOLERANCE:
                last_stable_confidence_time = now 
                stable_confidence_base = main_conf_score 
                
            # --- Б. Құлауды тексеру (Fall Detection) ---
            # --- ЖАҢАРТЫЛҒАН СТРОГАЯ ЛОГИКА (Aspect Ratio + Угол + Нос) ---

            is_lying_horizontal = False
            
            # ДАННЫЕ О ПОЛОЖЕНИИ ТЕЛА
            # Убедимся, что landmarks доступны
            if len(landmarks) >= 13: 
                try:
                    # Нормализованная позиция Носа (0 = верх, 1 = низ экрана)
                    nose_y_norm = landmarks[0][1] / h 
                    body_angle = compute_body_angle(landmarks, h, w) # Угол 0 = горизонтально, 90 = вертикально
                except:
                    nose_y_norm = 0
                    body_angle = 90
            else:
                nose_y_norm = 0
                body_angle = 90

            # 1. Проверка Aspect Ratio (Геометрия)
            if len(res.boxes.xyxy) > 0:
                box = res.boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = box
                w_box = x2 - x1
                h_box = y2 - y1
                aspect_ratio = w_box / h_box
                
                # Порог Ratio увеличен до 1.6, чтобы исключить сидение
                if aspect_ratio > 1.6: # <-- Самое строгое условие для лежания
                    is_lying_horizontal = True

            # 2. Проверка Угла Тела (Поза)
            # Тело должно быть наклонено не более чем на 35 градусов от горизонтали
            is_body_tilted_low = (body_angle is not None) and (body_angle > 70.0) # <-- Снижено с 45.0 до 35.0

            # 3. Проверка Низкого Носа (Исключает наклон над столом)
            # Если нос находится ниже 60% высоты экрана (0.6)
            is_nose_low = nose_y_norm > 0.6 
            
            # Окончательное решение: FALL DETECTED, если ЛЕЖИТ и находится НИЗКО 
            # (Комбинация Ratio, Угла и Носа)
            if is_lying_horizontal and is_body_tilted_low and is_nose_low:
                fall_detected = True
            else:
                fall_detected = False

            # --- ДИАГНОСТИКА ДЛЯ ЭКРАНА (Оставьте, пока не будете уверены) ---
            if 'aspect_ratio' in locals():
                cv2.putText(annotated_frame, f"Ratio: {aspect_ratio:.2f}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if body_angle is not None:
                cv2.putText(annotated_frame, f"Angle: {body_angle:.1f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(annotated_frame, f"Nose Y: {nose_y_norm:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # ---------------------------------------------------------------------

                if len(hip_history) >= 2:
                    hip_delta = hip_history[-1] - hip_history[0]
                    angle_delta = abs(angle_history[-1] - angle_history[0])
                    nose_low = nose_history[-1] > NOSE_LOW_THRESHOLD

                    if (hip_delta > HIP_DROP_THRESHOLD) and nose_low and (angle_delta > ANGLE_CHANGE_THRESHOLD):
                        fall_detected = True

            if fall_detected:
                fall_counter += 1
            else:
                fall_counter = 0

            # --- Құлау туралы хабарлама (ДВУХУРОВНЕВАЯ СИСТЕМА) ---
            
            # --- Сброс флага, если человек встал ---
            # --- Сброс флага, если человек встал ---
            if not fall_detected:
                if is_initial_alert_sent:
        
                    # 1. Формирование имени файла и сохранение текущего кадра
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    img_name = f"person_stood_up_{timestamp}.jpg"
                    cv2.imwrite(img_name, frame)
        
                    # 2. Вывод в консоль и формирование сообщения для Telegram
                    print(f"✅ Человек встал. Сброс тревоги.")
                    msg = f"✅ ЧЕЛОВЕК ВСТАЛ. Сброс тревоги.\nУақыты: {time.strftime('%H:%M:%S')}"
        
                    # 3. ОТПРАВКА ОПОВЕЩЕНИЯ В TELEGRAM
                    send_telegram_alert(img_name, msg) 

                    # 4. Сброс флагов и счетчиков
                    is_initial_alert_sent = False
                    is_critical_alert_sent = False  
                    last_initial_fall_time = 0
                    fall_counter = 0
            # 1. Засечено первое падение (fall_counter достигает MIN_CONSECUTIVE)
            if fall_counter >= MIN_CONSECUTIVE:
                
                if last_initial_fall_time == 0:
                    # Запись времени первого обнаружения
                    last_initial_fall_time = now
                
                # --- УРОВЕНЬ 1: Первое оповещение (Обнаружено падение) ---
                # Отправляем, если: не отправлено И Кулдаун прошел
                if not is_initial_alert_sent and (now - last_fall_alert_time) > COOLDOWN_SECONDS:
                    
                    last_fall_alert_time = now
                    is_initial_alert_sent = True 
                    
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    img_name = f"fall_initial_{timestamp}.jpg"
                    cv2.imwrite(img_name, frame)
                    
                    print(f"🚨 УРОВЕНЬ 1: Первичное обнаружение падения! {img_name}")
                    msg = f"⚠️ ОБНАРУЖЕНО ПАДЕНИЕ!\nПроверьте ситуацию.\nУақыты: {time.strftime('%H:%M:%S')}"
                    send_telegram_alert(img_name, msg)
                
                # --- УРОВЕНЬ 2: Критическая помощь (Падение продолжается) ---
                # Отправляем, если: Первое отправлено И Падение длится > FALL_STAY_THRESHOLD
                time_lying = now - last_initial_fall_time
                if is_initial_alert_sent and time_lying >= FALL_STAY_THRESHOLD:
                    
                    # Отправляем только один раз, пока человек не встанет
                    if not is_critical_alert_sent or ((now - last_fall_alert_time) > ALERT_COOLDOWN_LONG):
                        
                        last_fall_alert_time = now # Используем длинный кулдаун для повторных критических
                        is_critical_alert_sent = True
                        
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        img_name = f"fall_critical_{timestamp}.jpg"
                        cv2.imwrite(img_name, frame)
                        
                        print(f"🆘 УРОВЕНЬ 2: КРИТИЧЕСКАЯ СИТУАЦИЯ! {time_lying:.0f} секунд.")
                        msg = f"🆘 КРИТИЧЕСКАЯ СИТУАЦИЯ! 103\nПадение длится {time_lying:.0f} секунд!\nТРЕБУЕТСЯ ПОМОЩЬ!\nУақыты: {time.strftime('%H:%M:%S')}"
                        send_telegram_alert(img_name, msg)
            
            # Если fall_detected == True, fall_counter уже увеличился выше

    # 4. Логика: Адам ЖОҚ (No Person Logic)
    else:
        # Адам жоқ болса, Confidence Score-ды бастапқы күйге келтіру
        stable_confidence_base = 0.0 
        
        if (now - last_person_detection_time) > NO_PERSON_THRESHOLD_SECONDS:
            no_person_detected = True

    # ============================================================
    # ========== ЕСКЕРТУЛЕР (NO PERSON) =============
    # ============================================================
    
    if fall_counter < MIN_CONSECUTIVE:
        #fall_counter = 0

        # Сценарий: Адам мүлдем жоқ (No Person)
        if no_person_detected:
            status_text = "NO PERSON DETECTED"
            status_color = (255, 0, 255) 
            
            if (now - last_long_alert_time) > ALERT_COOLDOWN_LONG and not is_no_person_alert_sent:
                last_long_alert_time = now
                is_no_person_alert_sent = True 
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_name = f"noperson_{timestamp}.jpg"
                cv2.imwrite(img_name, frame)
                
                print("🚫 АДАМ ЖОҚ!")
                msg = f"🚫 ЕСКЕРТУ: ОРЫН БОС!\nКамерада {NO_PERSON_THRESHOLD_SECONDS} секундтан астам уақыт ешкім жоқ.\nУақыты: {time.strftime('%H:%M:%S')}"
                send_telegram_alert(img_name, msg)
        
        # Егер статус OK
        elif status_text == "Status: OK":
            if is_person_visible:
                 status_text = "Status: OK"

    # ------------------- Экранға жазу -------------------
    if fall_counter >= MIN_CONSECUTIVE:
        status_text = "FALL DETECTED!"
        status_color = (0, 0, 255)

    cv2.putText(annotated_frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    
    # Таймерлерді көрсету
    # ✅ Confidence Score-ға қатысты мәлімет жойылды
    timer_text = f"Gone: {int(now - last_person_detection_time)}s (Tol: {NO_PERSON_THRESHOLD_SECONDS}s)"
        
    cv2.putText(annotated_frame, timer_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Нәтижені көрсету
    cv2.imshow("Security System v10 - Display Cleaned", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == 27: # ESC басу
        break
    
    # FPS шектеу
    elapsed = time.time() - start_time
    if elapsed < (1.0 / TARGET_FPS):
        time.sleep((1.0 / TARGET_FPS) - elapsed)

cap.release()
cv2.destroyAllWindows()