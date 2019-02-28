from unittest import TestCase

import neuro_morpho_toolbox

class TestHelloWorld(TestCase):
    def test_is_string(self):
        s = neuro_morpho_toolbox.helloworld()
        self.assertTrue(isinstance(s, str))
