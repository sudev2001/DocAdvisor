# Generated by Django 4.2.7 on 2023-11-12 18:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chatbot", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Review",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("department", models.CharField(blank=True, max_length=100, null=True)),
            ],
        ),
    ]
