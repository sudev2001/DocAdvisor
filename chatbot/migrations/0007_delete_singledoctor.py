# Generated by Django 4.2.7 on 2023-12-01 10:04

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('chatbot', '0006_rename_multipledetails_multipledoctors'),
    ]

    operations = [
        migrations.DeleteModel(
            name='SingleDoctor',
        ),
    ]