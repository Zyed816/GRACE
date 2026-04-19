from django.core.management.base import BaseCommand

from lab.services import execute_experiment


class Command(BaseCommand):
    help = "Execute a queued experiment run in a separate worker process."

    def add_arguments(self, parser):
        parser.add_argument("run_id", type=int)

    def handle(self, *args, **options):
        execute_experiment(options["run_id"])
        self.stdout.write(self.style.SUCCESS(f"Experiment {options['run_id']} finished."))
