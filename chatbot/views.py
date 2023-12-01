from sentence_transformers import SentenceTransformer, util
import json
import os
import pandas as pd
import google.generativeai as palm
from typing import Any
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from .models import *
import nltk
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from django.db import connection
from django.core.serializers import serialize
from django.views.decorators.csrf import csrf_exempt
from .helpers import *

# Create your views here.


def _user_id(request):
    user = request.session.session_key
    if not user:
        user = request.session.create()
    return user


def Homepage(request):
    context = {
        'departments': DoctorDetails.objects.values_list('department', flat=True).distinct()
    }
    return render(request, 'Homepage.html', context)


def review_entry(request):
    if request.method == 'POST':
        department = request.POST.get('department')
        doctorname = request.POST.get('doctorname')
        review = request.POST.get('review')
        if not department.strip() or not doctorname.strip() or not review.strip():
            messages.error(request, "Fields are empty")
            return redirect('review_entry')
        try:
            doctor = DoctorDetails.objects.get(doctorname=doctorname)
            doctor_id = doctor.id
            Review.objects.create(department=department,
                                  doctor_id=doctor, review=review)
            print(doctor_id, 'ididididid---->')
            sentiment_analysis_palm(doctor_id, review)
            messages.success(request, "Review added successfully")
            return redirect('Homepage')
        except:
            messages.error(request, "Something went wrong")
            return redirect('review_entry')
    context = {
        'departments': DoctorDetails.objects.values_list('department', flat=True).distinct()
    }
    return render(request, 'review_form.html', context)


def get_doctors_by_department(request):
    selected_department = request.GET.get('department')
    doctors = DoctorDetails.objects.filter(
        department=selected_department).values('id', 'doctorname')
    doctors_list = list(doctors)
    return JsonResponse({'doctors': doctors_list}, safe=False)


def sentiment_analysis(id, review):
    lemma = WordNetLemmatizer()

    tfidf_vectorizer = joblib.load('datasets/tfidf_vectorizer.joblib')
    naive = joblib.load('datasets/naive.joblib')

    review = re.sub(r'<[^>]+>', '', review)
    # remove non alphanumeric characters
    review = re.sub('[^a-zA-Z0-9]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [lemma.lemmatize(
        word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)

    tfidf_matrix = tfidf_vectorizer.transform([review])

    sentiment = naive.predict(tfidf_matrix)
    # df.loc[len(df)-1,['sentiment']] = sentiment
    try:
        doc = DoctorDetails.objects.get(id=id)

        if sentiment == [-1]:
            doc.negative_count += 1
            # doc.loc[id,['negative_count']]+=1

        elif sentiment == [0]:
            doc.neutral_count += 1
            # doc.loc[id,['neutral_count']]+=1

        else:
            doc.positive_count += 1
            # doc.loc[id,['positive_count']]+=1

        doc.total_reviews += 1
        # doc.loc[id,['total_reviews']]+=1

        doc.rank_math = (doc.positive_count -
                         doc.negative_count)/doc.total_reviews
        # doc.loc[id,['rank_math']]= (doc.loc[id]['positive_count'] - doc.loc[id]['negative_count'])/doc.loc[id]['total_reviews']

        doc.save()

        # Define the raw SQL query with the correct table name
        sql_query = """
        WITH ranked_doctors AS (
            SELECT
                doctorname,
                department,
                experience,
                positive_count,
                negative_count,
                neutral_count,
                total_reviews,
                rank_math,
                RANK() OVER (PARTITION BY department ORDER BY rank_math DESC) AS department_rank
            FROM
                chatbot_doctordetails
        )
        UPDATE chatbot_doctordetails AS d
        SET rank = r.department_rank
        FROM ranked_doctors r
        WHERE d.doctorname = r.doctorname AND d.department = r.department;
        """

        # Execute the raw SQL query
        with connection.cursor() as cursor:
            cursor.execute(sql_query)

        # # dept_grp = doc.groupby('Department')
        # # doc['Rank'] = dept_grp.rank_math.rank(ascending=False).astype('int')
    except:
        pass


# Palm_key = os.environ.get('Palm_key')
palm.configure(api_key='AIzaSyCpB0Nrq168cfAaDLNYtY_r-9VxPie2OeE')
model_id = 'models/text-bison-001'


def sentiment_analysis_palm(id, review):
    promt = '''
        Conduct a sentiment analysis on the sentence. Assign a score of 1 for positivity, -1 for negativity, or 0 for neutrality based on the analysis results.
    '''
    completion = palm.generate_text(
        model=model_id,
        prompt=f"{review}\n{promt}",
        temperature=0.0,
        max_output_tokens=1600,
        candidate_count=1)
    result = int(completion.result)
    print(result, '____sudev____')
    try:
        doc = DoctorDetails.objects.get(id=id)

        if result == -1:
            doc.negative_count += 1
            # doc.loc[id,['negative_count']]+=1

        elif result == 0:
            doc.neutral_count += 1
            # doc.loc[id,['neutral_count']]+=1

        else:
            doc.positive_count += 1
            # doc.loc[id,['positive_count']]+=1

        doc.total_reviews += 1
        # doc.loc[id,['total_reviews']]+=1

        doc.rank_math = (doc.positive_count -
                         doc.negative_count)/doc.total_reviews
        # doc.loc[id,['rank_math']]= (doc.loc[id]['positive_count'] - doc.loc[id]['negative_count'])/doc.loc[id]['total_reviews']

        doc.save()

        # Define the raw SQL query with the correct table name
        sql_query = """
            WITH ranked_doctors AS (
                SELECT
                    doctorname,
                    department,
                    experience,
                    place,
                    hospital_name,
                    contact_number,
                    positive_count,
                    negative_count,
                    neutral_count,
                    total_reviews,
                    rank_math,
                    RANK() OVER (PARTITION BY hospital_name, department ORDER BY rank_math DESC) AS hospital_department_rank
                FROM
                    chatbot_doctordetails
            )
            UPDATE chatbot_doctordetails AS d
            SET rank = r.hospital_department_rank
            FROM ranked_doctors r
            WHERE d.doctorname = r.doctorname AND d.department = r.department;
        """

        # Execute the raw SQL query
        with connection.cursor() as cursor:
            cursor.execute(sql_query)

        # # dept_grp = doc.groupby('Department')
        # # doc['Rank'] = dept_grp.rank_math.rank(ascending=False).astype('int')
    except:
        pass


@csrf_exempt
def user_question(request):
    question = request.POST.get('question')
    print(question)
    answer = request.POST.get('answer')
    user = _user_id(request)

    data1 = DoctorDetails.objects.all()
    data = pd.DataFrame.from_records(data1.values())

    if not question or question == 'Please enter the dept name properly.':
        response = find_department(answer, data, user)
        return JsonResponse({'response': response})
    elif question:
        if question in ['If you require place, please enter the place else NO', 'Please enter the place properly.']:
            response = find_place(answer, data, user)
            return JsonResponse({'response': response})
        elif question.startswith('If require hospital, please type hospital name') or question== 'Please enter the hospital name properly':
            response = find_hospital(answer, data, user)
            return JsonResponse({'response': response})
        elif question in ['If require experience, please enter experience from 1 to 5 (in Years) else NO', 'Please enter the experience in proper range']:
            response = experience(answer, data, user)
            return JsonResponse({'response': response})

    return JsonResponse({'response': answer})


def ask_me(request):
    return render(request, 'bot.html')


def get_doctor_details(request, id):
    doctor = DoctorDetails.objects.get(id=id)
    serialized_doctor = serialize('json', [doctor])
    data = json.loads(serialized_doctor)

    # Access the fields dictionary
    fields = data[0]['fields']
    # print(fields)
    return JsonResponse({'doctor': fields}, safe=False)
