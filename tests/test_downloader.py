"""Unit tests for src.downloader."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.downloader import (
    _normalize_dataframe,
    get_ticker_csv_path,
    load_ticker_data,
    save_ticker_data,
)


class TestDownloader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalize_dataframe(self):
        # Create tz-aware index with non-midnight times
        dates = pd.date_range("2026-01-01 09:30:00", periods=3, tz="America/New_York")
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates)

        normalized = _normalize_dataframe(df)
        self.assertIsNone(normalized.index.tz)
        self.assertEqual(normalized.index.name, "Date")
        self.assertEqual(str(normalized.index[0]), "2026-01-01 00:00:00")

    @patch("src.downloader.yf.Ticker")
    def test_save_ticker_data_fresh(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_ticker_cls.return_value = mock_instance

        dates = pd.date_range("2026-01-01", periods=3, tz="UTC")
        sample_df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [105.0, 106.0, 107.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [104.0, 105.0, 106.0],
                "Adj Close": [104.0, 105.0, 106.0],
                "Volume": [1000, 1100, 1200],
            },
            index=dates,
        )
        mock_instance.history.return_value = sample_df

        saved_path = save_ticker_data(
            ticker="AAPL",
            data_dir=self.data_dir,
            period="1mo",
            incremental=True,
        )

        expected_file = self.data_dir / "AAPL.csv"
        self.assertEqual(saved_path, expected_file)
        self.assertTrue(expected_file.exists())

        loaded_df = load_ticker_data("AAPL", data_dir=self.data_dir)
        self.assertEqual(len(loaded_df), 3)
        self.assertIn("Close", loaded_df.columns)
        self.assertIn("Adj Close", loaded_df.columns)
        self.assertEqual(str(loaded_df.index[0].date()), "2026-01-01")

    @patch("src.downloader.yf.Ticker")
    def test_save_ticker_data_incremental_merge(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_ticker_cls.return_value = mock_instance

        # Initial save (Day 1 and Day 2)
        initial_dates = pd.date_range("2026-01-01", periods=2, tz="UTC")
        initial_df = pd.DataFrame(
            {
                "Open": [100.0, 105.0],
                "High": [102.0, 107.0],
                "Low": [98.0, 103.0],
                "Close": [101.0, 106.0],
                "Adj Close": [101.0, 106.0],
                "Volume": [1000, 1500],
            },
            index=initial_dates,
        )
        mock_instance.history.return_value = initial_df

        save_ticker_data("MSFT", data_dir=self.data_dir)

        # Second update: Day 2 (revised close) and Day 3 (new)
        update_dates = pd.date_range("2026-01-02", periods=2, tz="UTC")
        update_df = pd.DataFrame(
            {
                "Open": [105.0, 108.0],
                "High": [107.0, 110.0],
                "Low": [103.0, 107.0],
                "Close": [106.5, 109.0],  # Updated Day 2 close from 106.0 to 106.5
                "Adj Close": [106.5, 109.0],
                "Volume": [1600, 2000],
            },
            index=update_dates,
        )
        mock_instance.history.return_value = update_df

        save_ticker_data("MSFT", data_dir=self.data_dir, incremental=True)

        final_df = load_ticker_data("MSFT", data_dir=self.data_dir)
        # Should have 3 total unique rows (Jan 1, Jan 2, Jan 3)
        self.assertEqual(len(final_df), 3)
        # Jan 2 Close should be the updated 106.5
        jan_2_ts = pd.Timestamp("2026-01-02")
        self.assertEqual(final_df.loc[jan_2_ts, "Close"], 106.5)

    @patch("src.downloader.yf.Ticker")
    def test_empty_result_raises_error_on_fresh(self, mock_ticker_cls):
        mock_instance = MagicMock()
        mock_ticker_cls.return_value = mock_instance
        mock_instance.history.return_value = pd.DataFrame()

        with self.assertRaises(ValueError):
            save_ticker_data("INVALID", data_dir=self.data_dir)


if __name__ == "__main__":
    unittest.main()
