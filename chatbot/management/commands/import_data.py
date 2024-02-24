# import_data.py
import csv
from django.core.management.base import BaseCommand
from chatbot.models import DoctorDetails

class Command(BaseCommand):
    help = 'Import data from CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                DoctorDetails.objects.create(
                    doctorname=row['doctor_name'],
                    department=row['department'],
                    experience=row['experience'],
                    place=row['place'],
                    hospital_name=row['hospital_name'],
                    contact_number=row['contact_number'],
                    positive_count=row['positive_count'],
                    negative_count=row['negative_count'],
                    neutral_count=row['neutral_count'],
                    total_reviews=row['total_reviews'],
                    rank_math=row['rank_math'],
                    rank=row['rank'],
                    
                )
        self.stdout.write(self.style.SUCCESS('Data imported successfully'))
