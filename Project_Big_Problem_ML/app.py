import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, Normalizer, LabelEncoder
from sklearn.cluster import KMeans
import psycopg2
from psycopg2 import Error
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from datetime import datetime, time, date
from sklearn.metrics import r2_score
import pickle
from sklearn.preprocessing import LabelEncoder


def pred(df, cl_cl, lev_cl, level_en, scaler_cl, scaler):
    dd = df.copy()
    from sklearn.utils import column_or_1d
    class ChEncoder(LabelEncoder):
        def fit(self, y):
            y = column_or_1d(y, warn=True)
            self.classes_ = pd.Series(y).unique()
            return self
    y_n_en = ChEncoder()
    y_n_en.fit(["Нет", "Да"])
    for i in dd.columns[3:]:
        dd[i] = y_n_en.transform(dd[i])
        dd[i] += 1
    
    Xc = scaler_cl.transform(dd.drop(["patient_id", "gender"], axis=1))
    dd["clust"] = cl_cl.predict(Xc)
    X = scaler.transform(dd.drop(["patient_id", "gender"], axis=1))
    dd.drop(["clust"], axis=1, inplace=True) 
    for i in dd.columns[3:]:
        dd[i] -= 1
        dd[i] = y_n_en.inverse_transform(dd[i])
    dd["lung_cancer"] = lev_cl.predict(X)
    
   
    dd["lung_cancer"] = level_en.inverse_transform(dd["lung_cancer"])
    return dd

def pred1(df, cl_cl, lev_cl,  level_en, scaler_cl, scaler):
    dd = df.copy()
    gender_en = LabelEncoder()
    dd["Gender"] = gender_en.fit_transform(dd["Gender"])
    print(dd["Gender"].unique())
    Xc = scaler_cl.transform(dd.drop(["Patient Id"], axis=1))
    dd["clust"] = cl_cl.predict(Xc)
    X = scaler.transform(dd.drop(["Patient Id"], axis=1))
    dd["Level"] = lev_cl.predict(X)
    dd.drop(["clust"], axis=1, inplace=True)
    dd["Gender"] = gender_en.inverse_transform(dd["Gender"])
    dd["Level"] = level_en.inverse_transform(dd["Level"])
    return dd



# "patient_id", "gender", "age", "smoking", "yellow_fingers", "anxiety", "peer_pressure", "chronic_disease", "fatigue", "allergy", "wheezing", "alcohol", "coughing", "shortness_of_breath", "swallowing_difficulty", "chest_pain"
def new_input_y_n(nb, pi, ge, ag, sm, yf, an, pp, cd, fa, al, wh, alc, cou, sob, sd, cp, count):
    # Создание переменной, для составления уникального ключа каждому интерактивному элементу
    count *= 25
    
    with pi:
        nb["patient_id"] += [st.number_input("Введите ID пациента: ", min_value=0, key=count-113)]

    with ge:
        nb["gender"] += [st.selectbox("Пол", ["Мужчина", "Женщина"],  key=count-114)]
    
    with ag:
        nb["age"] += [st.number_input("Сколько вам лет: ", min_value=0, key=count-112)]
    
    with sm:
        nb["smoking"] += [st.selectbox("Вы курите", ["Нет", "Да"],  key=count-111)]
    
    with yf:
        nb["yellow_fingers"] += [st.selectbox("Жёлтые пальцы", ["Нет", "Да"], help="Случай паронихии, инфицирования складок тканей вокруг ногтей. Обычно прилегающие ткани болезненны, краснеют и опухли. Может быть гной, который обычно желтого цвета, отсюда и название.",  key=count-110)]
    
    with an:
        nb["anxiety"] += [st.selectbox("Испытываете тревожность", ["Нет", "Да"],  key=count-109)]
    
    with pp:
        nb["peer_pressure"] += [st.selectbox("Испытываете эмоциональное давление", ["Нет", "Да"],  key=count-108)]
   
    with cd:
        nb["chronic_disease"] += [st.selectbox("Имеете хронические заболевания", ["Нет", "Да"],  key=count-107)]
    
    with fa:
        nb["fatigue"] += [st.selectbox("Чувствуете усталость", ["Нет", "Да"],  key=count-106)]

    with al:
        nb["allergy"] += [st.selectbox("Есть аллергия", ["Нет", "Да"],  key=count-105)]

    with wh:
        nb["wheezing"] += [st.selectbox("Сопите", ["Нет", "Да"],  key=count-104)]

    with alc:
        nb["alcohol"] += [st.selectbox("Пьёте алкоголь", ["Нет", "Да"],  key=count-103)]

    with cou:
        nb["coughing"] += [st.selectbox("Кашель", ["Нет", "Да"],  key=count-102)]

    with sob:
        nb["shortness_of_breath"] += [st.selectbox("Одышка", ["Нет", "Да"],  key=count-101)]

    with sd:
        nb["swallowing_difficulty"] += [st.selectbox("Трудности с глотанием", ["Нет", "Да"],  key=count-100)]
    
    with cp:
        nb["chest_pain"] += [st.selectbox("Ощущаете боли в груди", ["Нет", "Да"],  key=count-99)]



def main(): 
    # Установка ширины страницы
    st.set_page_config(layout="wide")

    # Кнопка для обновления страницы без потери значений
    st.button("Обновить", help="Обновить страницу")

    # Название
    st.title("Предсказание рака лёгких") 
            
    # Другое название
    html_temp = """ 
    <div style ="background-color:blue;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Streamlit ML App </h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True) 

    if st.toggle("Да_Нет/Слайдер"):
        with open("Project_slider/label_encoder_level.pickle", "rb") as file:
            cancer_le = pickle.load(file)
        
        with open("Project_slider/model_rfc_clust.pkl", "rb") as file:
            rfc_clust = pickle.load(file)

        if "RandomForestClassifier" not in st.session_state:
            with open("Project_slider/model_rfc.pkl", "rb") as file:
                st.session_state["RandomForestClassifier"] = pickle.load(file)
        if "ExtraTreesClassifier" not in st.session_state:
            with open("Project_slider/model_etc.pkl", "rb") as file:
                st.session_state["ExtraTreesClassifier"] = pickle.load(file)
        if "GradientBoostingClassifier" not in st.session_state:
            with open("Project_slider/model_gbc.pkl", "rb") as file:
                st.session_state["GradientBoostingClassifier"] = pickle.load(file)
        if "HistGradientBoostingClassifier" not in st.session_state:
            with open("Project_slider/model_hgbc.pkl", "rb") as file:
                st.session_state["HistGradientBoostingClassifier"] = pickle.load(file)
        if "AdaBoostClassifier" not in st.session_state:
            with open("Project_slider/model_adc.pkl", "rb") as file:
                st.session_state["AdaBoostClassifier"] = pickle.load(file)
        if "PassiveAggressiveClassifier" not in st.session_state:
            with open("Project_slider/model_pac.pkl", "rb") as file:
                st.session_state["PassiveAggressiveClassifier"] = pickle.load(file)

        with open("Project_slider/scaler_clust.pkl", "rb") as file:
            scaler_cl = pickle.load(file)
        with open("Project_slider/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)

        nb = {}
            
        # Базы данных
        new_base = pd.DataFrame()

        flag2=False
        # Возможность выбора ввода
        if st.toggle("Загрузить файл / ввести вручную"):
            flag2=True
            l_col = ['Patient Id', 'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring']
            nb = {i:[] for i in l_col}
            

            nb["Patient Id"] += [st.number_input("Введите ID пациента: ", min_value=0)]

            nb["Age"] += [st.number_input("Сколько вам лет: ", min_value=1)]

            nb["Gender"] += [st.selectbox("Пол", ["Мужчина", "Женщина"])]

            st.markdown("Введите тяжесть симптома, где левая часть - не имею, правая - тяжело проявляется.")

            nb["Air Pollution"] += [st.slider("Загрязнение воздуха вокруг", min_value=1, max_value=8)]

            nb["Alcohol use"] += [st.slider("Престрастие к алкоголю", min_value=1, max_value=8)]

            nb["Dust Allergy"] += [st.slider("Аллергия на пыль", min_value=1, max_value=8)]

            nb["OccuPational Hazards"] += [st.slider("Опасное окружение для здоровья на работе", min_value=1, max_value=8)]

            nb["Genetic Risk"] += [st.slider("Генетический риск", min_value=1, max_value=7)]

            nb["chronic Lung Disease"] += [st.slider("Хронические заболевания лёгких", min_value=1, max_value=7)]

            nb["Balanced Diet"] += [st.slider("Сбалансированная диета", min_value=1, max_value=7)]

            nb["Obesity"] += [st.slider("Ожирение", min_value=1, max_value=7)]

            nb["Smoking"] += [st.slider("Курение", min_value=1, max_value=8)]

            nb["Passive Smoker"] += [st.slider("Пассивное курение", min_value=1, max_value=8)]

            nb["Chest Pain"] += [st.slider("Боли в груди", min_value=1, max_value=9)]

            nb["Coughing of Blood"] += [st.slider("Кашель с кровью", min_value=1, max_value=9)]

            nb["Fatigue"] += [st.slider("Усталость", min_value=1, max_value=9)]

            nb["Weight Loss"] += [st.slider("Потеря веса", min_value=1, max_value=8)]

            nb["Shortness of Breath"] += [st.slider("Одышка", min_value=1, max_value=9)]

            nb["Wheezing"] += [st.slider("Сопение", min_value=1, max_value=8)]

            nb["Swallowing Difficulty"] += [st.slider("Затруднение глотания", min_value=1, max_value=8)]

            nb["Clubbing of Finger Nails"] += [st.slider("Изменения в областях под и вокруг ногтей пальцев рук", min_value=1, max_value=9)]

            nb["Frequent Cold"] += [st.slider("Частота простудных заболеваний", min_value=1, max_value=7)]

            nb["Dry Cough"] += [st.slider("Сухой кашель", min_value=1, max_value=7)]

            nb["Snoring"] += [st.slider("Храп", min_value=1, max_value=7)]


            

            new_base = pd.DataFrame(nb)

            # Вывод получившейся базы
            if st.button("Вывести базу", help="Нажмите, чтобы закончить добавлять строки"):
                st.dataframe(new_base)
                pass
            
        else:
            # Загрузка файла
            new_base = st.file_uploader("Загрузите новые данные: ", type=["csv"], help="Загрузите сюда CSV-файл, состоящий из  следующих колонок: 'Patient Id', 'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy', 'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease', 'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker', 'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss', 'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty', 'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'")
        
        model = st.session_state[f'{st.selectbox("Выберите модель предсказания", ["RandomForestClassifier", "GradientBoostingClassifier", "HistGradientBoostingClassifier", "ExtraTreesClassifier", "AdaBoostClassifier", "PassiveAggressiveClassifier"])}']
        flag = False
        if flag2:
            if st.toggle("Вывести рекомендации"):
                flag = True
        if st.button("Предсказать"):
            new_base = pred1(new_base, rfc_clust, model, cancer_le, scaler_cl, scaler)
            
            ind = cancer_le.transform(new_base["Level"])
            l_bd = []
            l_d = ["Шанс рака высокий, как можно быстрее обратитесь к врачу", "Шанс рака минимальный", "Шанс рака имеется, советуем обратиться к врачу"]
            for i in ind:
                l_bd.append(l_d[i])
            new_base["Recomendation"] = l_bd
            st.dataframe(new_base)
            if flag:
                for index, i in new_base.iterrows():
                    st.markdown(f"### Пациент {i['Patient Id']}: {i['Recomendation']}")


    else:
        with open("Project_Yes_No/label_encoder_cancer.pickle", "rb") as file:
            cancer_le = pickle.load(file)
        
        with open("Project_Yes_No/model_rfc_clust.pkl", "rb") as file:
            rfc_clust = pickle.load(file)
        with open("Project_Yes_No/model_rfc.pkl", "rb") as file:
            rfc = pickle.load(file)

        with open("Project_Yes_No/scaler_clust.pkl", "rb") as file:
            scaler_cl = pickle.load(file)
        with open("Project_Yes_No/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)

        
            
        # Базы данных
        new_base = pd.DataFrame()


        # Возможность выбора ввода
        if st.toggle("Загрузить файл / ввести вручную"):
            # Разделение по колонкам
            
            nb = {"patient_id":[], "gender":[], "age":[], "smoking":[], "yellow_fingers":[], "anxiety":[], "peer_pressure":[], "chronic_disease":[], "fatigue":[], "allergy":[], "wheezing":[], "alcohol":[], "coughing":[], "shortness_of_breath":[], "swallowing_difficulty":[], "chest_pain":[]}
            # Выбор кол-ва строк ввода
            num_rows = st.slider("Выберите ко-во строк: ", min_value=1, max_value=10)
            # Ввод строк
            for i in range(1, num_rows+1):
                pi, ge, ag, sm, yf, an, pp, cd, fa, al, wh, alc, cou, sob, sd, cp = st.columns(16, vertical_alignment="bottom")
                new_input_y_n(nb, pi, ge, ag, sm, yf, an, pp, cd, fa, al, wh, alc, cou, sob, sd, cp, i)
            
            new_base = pd.DataFrame(nb)

            # Вывод получившейся базы
            if st.button("Вывести базу", help="Нажмите, чтобы закончить добавлять строки"):
                st.dataframe(new_base)
                pass
            
        else:
            # Загрузка файла
            new_base = st.file_uploader("Загрузите новые данные: ", type=["csv"], help='Загрузите сюда CSV-файл, состоящий из  следующих колонок: "patient_id", "age", "smoking", "yellow_fingers", "anxiety", "peer_pressure", "chronic_disease", "fatigue", "allergy", "wheezing", "alcohol", "coughing", "shortness_of_breath", "swallowing_difficulty", "chest_pain"')
        
        if st.button("Предсказать"):
            new_base = pred(new_base, rfc_clust, rfc, cancer_le, scaler_cl, scaler)
            st.dataframe(new_base)


    
    

# Проверка на среду, сценарий и запуск
if __name__=='__main__': 
    main() 
