from pathlib import Path

from django.core.management.base import BaseCommand

from apps.experiments.services import sync_result_csv


class Command(BaseCommand):
    help = "Sync pipeline result CSV files under the GRACE results directory into the database."

    def handle(self, *args, **options):
        from django.conf import settings

        results_dir = Path(settings.GRACE_RESULTS_DIR)
        if not results_dir.exists():
            self.stdout.write(self.style.WARNING(f"Results directory not found: {results_dir}"))
            return

        total_files = 0
        total_rows = 0
        for csv_path in sorted(results_dir.glob("*_full_pipeline_results.csv")):
            synced = sync_result_csv(csv_path)
            total_files += 1
            total_rows += len(synced)
            self.stdout.write(self.style.SUCCESS(f"Synced {len(synced)} rows from {csv_path.name}"))

        self.stdout.write(self.style.SUCCESS(f"Finished syncing {total_rows} rows from {total_files} files."))