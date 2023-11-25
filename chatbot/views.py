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

# Create your views here.

def Homepage(request):
    context = {
        'departments' : DoctorDetails.objects.values_list('department', flat=True).distinct()
    }
    return render(request,'Homepage.html',context)

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
            Review.objects.create(department=department, doctor_id=doctor, review=review)
            sentiment_analysis_palm(doctor_id,review)
            messages.success(request,"Review added successfully")
            return redirect('Homepage')
        except:
            messages.error(request,"Something went wrong")
            return redirect('review_entry')
    context = {
        'departments' : DoctorDetails.objects.values_list('department', flat=True).distinct()
    }
    return render(request,'review_form.html',context)

def get_doctors_by_department(request):
    selected_department = request.GET.get('department')
    doctors = DoctorDetails.objects.filter(department=selected_department).values('id','doctorname')
    doctors_list = list(doctors)
    return JsonResponse({'doctors': doctors_list}, safe=False)

def sentiment_analysis(id,review):
    lemma = WordNetLemmatizer()
    
    tfidf_vectorizer = joblib.load('datasets/tfidf_vectorizer.joblib')
    naive = joblib.load('datasets/naive.joblib')
    
    review = re.sub(r'<[^>]+>', '', review)
    review = re.sub('[^a-zA-Z0-9]', ' ',review) ## remove non alphanumeric characters
    review = review.lower()
    review = review.split()
    review = [lemma.lemmatize(word) for word in review if not word in stopwords.words('english')]
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
            doc.neutral_count +=1
            # doc.loc[id,['neutral_count']]+=1

        else:
            doc.positive_count +=1
            # doc.loc[id,['positive_count']]+=1
        
        doc.total_reviews +=1
        # doc.loc[id,['total_reviews']]+=1

        doc.rank_math = (doc.positive_count - doc.negative_count)/doc.total_reviews
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

import os
import google.generativeai as palm

Palm_key = os.environ.get('Palm_key')
palm.configure(api_key=Palm_key)
model_id='models/text-bison-001'

def sentiment_analysis_palm(id,review):
    promt='''
        Do sentimental analysis of the sentence give 1 if it is positive or -1 if it is negative or 0 if it is neutral
        '''
    completion=palm.generate_text(
    model=model_id,
    prompt=f"{review}\n{promt}",
    temperature=0.0,
    max_output_tokens=1600,
    candidate_count=1)
    result=int(completion.result)
    print(result,'____sudev____')
    try:
        doc = DoctorDetails.objects.get(id=id)

        if result == -1:
            doc.negative_count += 1            
            # doc.loc[id,['negative_count']]+=1

        elif result == 0:
            doc.neutral_count +=1
            # doc.loc[id,['neutral_count']]+=1

        else:
            doc.positive_count +=1
            # doc.loc[id,['positive_count']]+=1
        
        doc.total_reviews +=1
        # doc.loc[id,['total_reviews']]+=1

        doc.rank_math = (doc.positive_count - doc.negative_count)/doc.total_reviews
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

from happytransformer import HappyTextToText, TTSettings
from sentence_transformers import SentenceTransformer, util
from pandasai import PandasAI
from pandasai.llm import Starcoder                      
import pandas as pd

@csrf_exempt
def user_question(request):
    question = request.POST.get('question')
    if not question: return JsonResponse({'error':'enter valid question'})
    user_input = question.strip()
    if not user_input: return JsonResponse({'error':'enter valid question'})

    reference_questions=["List the doctors in the Cardiology department.",
    "Give me a list of 5 doctors in the Ortho department with the highest rank and their experience.",
    "List the best doctor in Pediatrics with more than three years of experience and the best rank.",
    "Provide details for the top-ranked doctor in the ENT department.",
    "How many doctors in the General department have more than three years of experience?",
    "Who has more than three years of experience and the best rank in Cardiology?",
    "Who has the highest number of positive reviews among all departments?",
    "Give me the top 3 doctors in the Psychology department based on their rank.",
    "List all the available departments in the hospital.",
    "List the best doctor in Ortho with more than three years of experience and the best rank.",
    "How many doctors are there in the Cardiology department?",
    "Who has more than three years of experience in Ortho?",
    "Provide a list of 5 doctors in the Ortho department and their experience, having the best rank.",
    "Who has more than three years of experience in ENT?",
    "List the doctors in ENT with the best rank.",
    "Find the doctor with the fewest negative reviews in any department.",
    "Can you list doctors who specialize in the Psychology department?",
    "Give me the best doctors in ENT with the best rank.",
    "Show the departments.",
    "List the doctors in the General department with the best rank.",
    "Who is the best-rated doctor in the Psychology department with over three years of experience?",
    "How many doctors are there in the Cardiology department?",
    "Which doctor in Ortho has more than three years of experience?",
    "Who has more than three years of experience and the best rank in General?",
    "How many doctors are there in Cardiology?",
    "List the doctors who have a rank of 1 in any department.",
    "Who has more than three years of experience in Ortho department?",
    "How many doctors are there in Cardiology department?",
    "Who has the maximum number of positive reviews?",
    "Which doctor in ENT department has more than three years of experience?",
    'list the doctors in General department',
    'Give me a list of 5 doctors in the orthopedics department with the highest rank and their experience.',
    'list the best doctor in Ortho having more than three years experience and best rank',
    'Provide details for the top-ranked doctor in the pediatrics department.',
    'how many doctors in cardiology department have more than three years experience',
    'who have more than three years experience and best rank in General department',
    'Who has the highest number of positive reviews among all departments?',
    'Give me the top 3 doctors in the cardiology department based on their rank.List all the available departments in the hospital.',
    'list the best doctor in psychology having more than three years experience',
    'how many doctors are there in Cardiology department',
    'who have more than three years experience in Orhto',
    'provide a list of 5 doctors in ortho department and their experience having best rank',
    'who have more than three years experience in ENT department.',
    'List the doctors in ENT having best rank',
    'Find the doctor with the fewest negative reviews in any department.Can you list doctors who specialize in the psychology department?',
    'Give me best doctors in ENT having best rank',
    'show the departments',
    'list the doctors in General department having best rank',
    'Who is the best-rated doctor in the psychology department with over three years of experience?',
    'how many departments are there',
    'which doctor in Orhto have more than three years experience',
    'who have more than three years experience and best rank in General department.',
    'how many doctors are there in cardiology',
    'List the doctors who have a rank of 1 in any department.',
    'who have more than three years experience in Orhto department',
    'how many doctors are there in cardiology department',
    'who have maximum number of positive reviews?',
    'which doctor in ENT department have more than three years experience',
    'How many doctors in the cardiology department have more than five years of experience?',
    'who have more than three years experience in General department',
    'give me the doctors in cardiology having best rank.',
    'give me the departments in a list',
    'who have more than three years experience in ENT department?',
    'which doctor in ENT have more than three years experience',
    'provide a list of doctors who worked in psychology department',
    'List the doctors having the rank 1.',
    'Provide details for a doctor in the general department named Jason Rodriguez.',
    "list the best doctor in 'General' having more than three years experience and best rank",
    'list the departments',
    'show the number of doctors are there in Cardiology department',
    'who have more than three years experience and best rank in ENT department.',
    "Give me best doctors in 'ENT' having best rank",
    'Give me the names of doctors in the orthopedics department with no negative reviews.',
    "list the doctors in 'General' department",
    'What is the total number of doctors in the pediatrics department?',
    'can you please show the details of doctor Jason Rodriguez',
    'give me the rank of the doctor Jason Rodriguez',
    'How many doctors are there in the ortho department?',
    'who have more than three years experience and best rank in Pediatrics department.',
    'Give me best doctors in ENT department having best rank',
    'Find doctors in the general department with at least ten years of experience.',
    'list out the department',
    'list the best doctor in General department having more than three years experience and best rank',
    'Give me the doctor who has the minimum number of negative reviews?',
    'list the doctors in General department those who have best rank',
    'What is the rank of Dr. Jason Rodriguez in the general department?']

    # Step 2: Correct any spelling and grammar mistakes (using HappyTextToText)
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    args = TTSettings(num_beams=5, min_length=1)
    result = happy_tt.generate_text("grammar: " + user_input, args=args)
    corrected_user_input = result.text
    print(corrected_user_input)
    # Load the SentenceTransformer model for similarity calculation
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Calculate similarity scores with all reference questions
    user_input_embedding = model.encode(corrected_user_input, convert_to_tensor=True)
    reference_question_embeddings = model.encode(reference_questions, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(user_input_embedding, reference_question_embeddings)[0]


    for i, score in enumerate(similarity_scores):
        try:
            data1 = DoctorDetails.objects.all()
            data = pd.DataFrame.from_records(data1.values())
            print('----')
            print(score)
            if data is None or len(data) == 0:
                error_message = "Upload a data file to proceed."
                return JsonResponse({"error": error_message})
            
            if score > 0.1:  # 20% similarity threshold
                # Step 4: Forward the question to the question answering model (using PandasAI)
                API_key = "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"
                llm = Starcoder(api_token=API_key)
                pandas_ai = PandasAI(llm, conversational=False, verbose=True)
                response = pandas_ai.run(data, prompt=corrected_user_input)
                print('Response --> ',response)

                if isinstance(response, int) or isinstance(response, float):
                    response_html = str(response)
                elif isinstance(response, list):
                    response_html = response
                elif isinstance(response, pd.DataFrame):
                    response_html = response.to_html(classes='table table-bordered', index=False)
                elif isinstance(response, pd.Series):
                    response_df = pd.DataFrame({'': response.index, 'Count': response.values})
                    response_html = response_df.to_html(classes='table table-bordered', index=False)
                else:
                    response_html = str(response)
                print(response_html,end="1234567890")
                print(type(response),end="asdfghjkl")
                return JsonResponse({'response': response_html,'message':"hello"})

                # Exit the loop once a matching question is found
            else:
                error_message = "Your question doesn't meet the similarity threshold."
                return JsonResponse({'error':error_message})
        except:
            return JsonResponse({'error':'Something went wrong'})

def ask_me(request):    
    return render(request,'bot.html')

import json
def get_doctor_details(request, id):
    doctor = DoctorDetails.objects.get(id=id)
    serialized_doctor = serialize('json', [doctor])
    data = json.loads(serialized_doctor)

    # Access the fields dictionary
    fields = data[0]['fields']
    # print(fields)
    return JsonResponse({'doctor': fields}, safe=False)
