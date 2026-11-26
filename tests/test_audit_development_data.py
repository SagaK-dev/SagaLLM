import unittest
from scripts.audit_development_data import check, PINS


class PinTests(unittest.TestCase):
    def test_pins(self):
        for name, value in PINS.items():
            self.assertEqual(check(name, value), value)
            with self.assertRaises(ValueError):
                check(name, "0"*64)
