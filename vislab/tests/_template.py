import logging
import unittest
import test_context
import vislab.predict


class TestX(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dirname = vislab.util.cleardirs(
            test_context.temp_dirname + '/test_X')


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()
