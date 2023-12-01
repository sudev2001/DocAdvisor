import pandas as pd
from .models import *
import numpy as np
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def find_department(user_dept, data, user):
    if MultipleDoctors.objects.filter(user=user).exists():
        to_delete = MultipleDoctors.objects.filter(user=user)
        to_delete.delete()

    dept = data.department.unique()
    user_input_embedding = model.encode(user_dept, convert_to_tensor=True)
    reference_question_embeddings = model.encode(dept, convert_to_tensor=True)
    similar_score = [util.pytorch_cos_sim(user_input_embedding, reference_question_embeddings[i])[
        0].item() for i in range(len(dept))]
    print(max(similar_score))
    if max(similar_score) > .6:
        max_index = np.argmax(similar_score)
        MultipleDoctors.objects.create(user=user, department=dept[max_index])
        return "If you require place, please enter the place else NO"
    else:
        return "Please enter the dept name properly."


def find_place(user_place: str, data, user):
    place = data.place.unique()
    user_input_embedding = model.encode(user_place, convert_to_tensor=True)
    reference_question_embeddings = model.encode(place, convert_to_tensor=True)
    similar_score = [util.pytorch_cos_sim(user_input_embedding, reference_question_embeddings[i])[
        0].item() for i in range(len(place))]
    if user_place.lower() == 'no':
        return "If require experience, please enter experience from 1 to 5 (in Years) else NO"
    elif max(similar_score) > .6:
        max_index = np.argmax(similar_score)
        query = MultipleDoctors.objects.get(user=user)
        query.place = place[max_index]
        query.save()
        hsptl = data[data.place == place[max_index]]['hospital_name'].unique()
        return f"""If require hospital, please type hospital name {hsptl} else NO"""
    else:
        return "Please enter the place properly."

def find_hospital(user_hsptl,data,user):
    query = MultipleDoctors.objects.get(user=user)
    place = query.place
    hsptl = data[data.place == place]['hospital_name'].unique()
    if user_hsptl in hsptl:
        query.hospital_name = user_hsptl
        query.save()
        return "If require experience, please enter experience from 1 to 5 (in Years) else NO"
    else:
        return "Please enter the hospital name properly"

def experience(user_exp,data,user):
    query = MultipleDoctors.objects.get(user=user)
    if int(user_exp)>=1 and int(user_exp)<=5:
        if query.place is None:
            result = data[(data.department == query.department) & (data.experience>= int(user_exp))]
        elif query.place is not None and query.hospital_name is None:
            result = data[(data.department == query.department) & (data.place == query.place) & (data.experience>= int(user_exp))]
        else:
            result = data[(data.department == query.department) & (data.place == query.place) & (data.hospital_name == query.hospital_name) & (data.experience>= int(user_exp))]
        print('---------------------------------->>>>>>>>>>>>>>>>..')
        print(result)
        return result.to_html(classes='table table-bordered', index=False)
    else:
        return "Please enter the experience in proper range"