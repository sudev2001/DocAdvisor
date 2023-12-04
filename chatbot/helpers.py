import pandas as pd
from .models import *
import numpy as np


def find_department(user_dept, data, user):
    if MultipleDoctors.objects.filter(user=user).exists():
        to_delete = MultipleDoctors.objects.filter(user=user)
        to_delete.delete()

    dept = data.department.unique()
    dept1 = [val.lower() for val in dept]
    if user_dept.lower() in dept1:
        max_index = dept1.index(user_dept.lower())
        MultipleDoctors.objects.create(user=user, department=dept[max_index])
        return "If you require place, please enter the place else NO"
    else:
        return f"""Sorry. I couldn't understand the dept. Please select one of these dept. {dept}"""


def find_place(user_place: str, data, user):
    place = data.place.unique()
    place1 = [val.lower() for val in place]
    if user_place.lower() == 'no':
        return "If require experience, please enter experience from 1 to 5 (in Years) else NO"
    elif user_place.lower() in place1:
        max_index = place1.index(user_place.lower())
        query = MultipleDoctors.objects.get(user=user)
        query.place = place[max_index]
        query.save()
        hsptl = data[data.place == place[max_index]]['hospital_name'].unique()
        return f"""If require hospital, please type hospital name {hsptl} else NO"""
    else:
        return f"""Sorry. I couldn't understand the place. Please select one of these place. {place}"""

def find_hospital(user_hsptl,data,user):
    query = MultipleDoctors.objects.get(user=user)
    place = query.place
    hsptl = data[data.place == place]['hospital_name'].unique()
    hsptl1 = [hsptls.upper() for hsptls in hsptl]
    if user_hsptl.lower() == 'no':
        return 'If require experience, please enter experience from 1 to 5 (in Years) else NO'
    if user_hsptl.upper() in hsptl1:
        max_index = hsptl1.index(user_hsptl.upper())
        query.hospital_name = hsptl[max_index]
        query.save()
        return "If require experience, please enter experience from 1 to 5 (in Years) else NO"
    else:
        return "Please enter the hospital name properly"

def experience(user_exp,data,user):
    query = MultipleDoctors.objects.get(user=user)
    if user_exp.lower() == 'no':
        if query.place is None:
            result = data[(data.department == query.department)]
        elif query.place is not None and query.hospital_name is None:
            result = data[(data.department == query.department) & (data.place == query.place)]
        else:
            result = data[(data.department == query.department) & (data.place == query.place) & (data.hospital_name == query.hospital_name)]
    elif int(user_exp)>=1 and int(user_exp)<=5:
        if query.place is None:
            result = data[(data.department == query.department) & (data.experience>= int(user_exp))]
        elif query.place is not None and query.hospital_name is None:
            result = data[(data.department == query.department) & (data.place == query.place) & (data.experience>= int(user_exp))]
        else:
            result = data[(data.department == query.department) & (data.place == query.place) & (data.hospital_name == query.hospital_name) & (data.experience>= int(user_exp))]

    else:
        return "Please enter the experience in proper range"
    result.sort_values(by=['rank', 'experience'], ascending=[True, False], inplace=True)
    return result.to_dict(orient='records')
    