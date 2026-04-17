import time

from django.core.management.base import BaseCommand
from django.db import close_old_connections

from apps.experiments.models import Experiment
from apps.experiments.services import run_experiment


class Command(BaseCommand):
    help = "Process pending experiments in the database."

    def add_arguments(self, parser):
        parser.add_argument("--once", action="store_true", help="Process a single pending experiment and exit")
        parser.add_argument("--poll-interval", type=int, default=10)

    def handle(self, *args, **options):
        once = options["once"]
        interval = options["poll_interval"]

        while True:
            close_old_connections()
            experiment = Experiment.objects.filter(status=Experiment.STATUS_PENDING).order_by("created_time").first()
            if experiment:
                self.stdout.write(self.style.NOTICE(f"Running experiment {experiment.pk}..."))
                run_experiment(experiment)
            elif once:
                self.stdout.write(self.style.NOTICE("No pending experiments found."))
                return
            else:
                time.sleep(interval)
