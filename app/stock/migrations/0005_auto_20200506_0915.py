# Generated by Django 2.2.9 on 2020-05-06 01:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stock', '0004_asxindexdailyhistory_asxindexhistory'),
    ]

    operations = [
        migrations.AlterField(
            model_name='stockpricehistory',
            name='trade_value',
            field=models.DecimalField(decimal_places=3, default=0, max_digits=19),
        ),
    ]
