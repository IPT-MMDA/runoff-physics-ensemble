"""The quick-training cohort stays fixed and matches real map assignments."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import geopandas as gpd
from fastapi.testclient import TestClient
from shapely.geometry import Point, box

from complexity_ensemble.api import create_app
from complexity_ensemble.application import Application
from complexity_ensemble.hydropower_cohort import HYDROPOWER_50_BASINS, hydropower_training_cohort
from complexity_ensemble.regional import BasinCatalogRecord, CAMELSCHCatalog
from complexity_ensemble.server_settings import DatasetRegistration, HydropowerSettings, ServerSettings


class HydropowerCohortTests(unittest.TestCase):
    def setUp(self):
        self.records = [BasinCatalogRecord(basin, 'Gauge ' + basin, {'area': 86.4},
            eligible=True, geometry=box(i, 0, i + .8, .8)) for i, basin in enumerate(HYDROPOWER_50_BASINS)]
        self.catalog = CAMELSCHCatalog('.', self.records)
        self.sites = gpd.GeoDataFrame({'_site_id': list(HYDROPOWER_50_BASINS),
            '_site_name': ['Plant ' + b for b in HYDROPOWER_50_BASINS]},
            geometry=[Point(i + .4, .4) for i in range(50)], crs=4326)
        self.compatibility = {b: {mode: {'eligible': True, 'reasons': []}
            for mode in ('bafu_only', 'bafu_combined')} for b in HYDROPOWER_50_BASINS}

    def test_fixed_unique_50_each_have_assigned_plants(self):
        self.assertEqual(len(set(HYDROPOWER_50_BASINS)), 50)
        self.assertEqual(tuple(sorted(HYDROPOWER_50_BASINS)), HYDROPOWER_50_BASINS)
        for mode in ('none', 'bafu_only', 'bafu_combined'):
            result = hydropower_training_cohort(self.catalog, self.sites, mode=mode, compatibility=self.compatibility)
            self.assertTrue(result['available'])
            self.assertEqual(result['matched_plants'], 50)
            self.assertEqual(result['training_scope'], 'targets_only')
            self.assertFalse(result['date_coverage_checked'])
            self.assertTrue(all(r['matched_plants'] == 1 for r in result['catchments']))

    def test_missing_catchments_are_reported_without_substituting_ids(self):
        catalog = CAMELSCHCatalog('.', self.records[1:])
        result = hydropower_training_cohort(catalog, self.sites)
        self.assertFalse(result['available'])
        self.assertIn(HYDROPOWER_50_BASINS[0], result['unavailable_basins'])
        self.assertEqual(result['basin_ids'], list(HYDROPOWER_50_BASINS))

    def test_source_and_current_inventory_are_checked(self):
        basin = HYDROPOWER_50_BASINS[0]
        self.compatibility[basin]['bafu_only'] = {'eligible': False, 'reasons': ['Missing BAFU station']}
        result = hydropower_training_cohort(self.catalog, self.sites,
            mode='bafu_only', compatibility=self.compatibility)
        self.assertEqual(result['unavailable_basins'][basin], 'Missing BAFU station')
        result = hydropower_training_cohort(self.catalog, self.sites.iloc[1:])
        self.assertIn('No assigned', result['unavailable_basins'][basin])
        with self.assertRaises(ValueError):
            hydropower_training_cohort(self.catalog, self.sites, mode='unknown')

    def test_http_contract_authentication_and_full_resolution_geometry(self):
        with TemporaryDirectory() as root:
            settings = ServerSettings(Path(root) / 'state', {'ch': DatasetRegistration(Path(root) / 'data')},
                token='token', hydropower=HydropowerSettings(geopackage=Path(root) / 'missing.gpkg'))
            application = Application(settings); application.hydropower_sites = self.sites
            with TestClient(create_app(settings, application=application)) as client:
                self.assertEqual(client.get('/api/v1/datasets/ch/hydropower-cohort').status_code, 401)
                client.headers['Authorization'] = 'Bearer token'
                with patch('complexity_ensemble.application.CAMELSCHCatalog', return_value=self.catalog) as factory, \
                     patch('complexity_ensemble.bafu.basin_compatibility', return_value=self.compatibility):
                    response = client.get('/api/v1/datasets/ch/hydropower-cohort?mode=bafu_only')
                self.assertEqual(response.status_code, 200, response.text)
                self.assertTrue(response.json()['available'])
                factory.assert_called_once_with(Path(root) / 'data', simplify_geometry=False)
                self.assertEqual(client.get('/api/v1/datasets/unknown/hydropower-cohort').status_code, 404)


if __name__ == '__main__':
    unittest.main()
