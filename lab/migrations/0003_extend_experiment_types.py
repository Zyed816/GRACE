from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("lab", "0002_experimentrun_abort_and_worker_pid"),
    ]

    operations = [
        migrations.AlterField(
            model_name="experimentrun",
            name="experiment_type",
            field=models.CharField(
                choices=[
                    ("method_comparison", "方法比较流水线"),
                    ("sampling_bias", "采样偏差验证"),
                    ("sensitivity", "超参数敏感性分析"),
                    ("component_ablation", "组件级消融实验"),
                    ("efficiency", "效率实验"),
                    ("significance", "统计显著性实验"),
                ],
                max_length=32,
                verbose_name="实验类型",
            ),
        ),
    ]
