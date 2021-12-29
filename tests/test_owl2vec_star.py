#!/usr/bin/env python

"""Tests for `owl2vec_star` package."""


import unittest
from click.testing import CliRunner

from owl2vec_star import owl2vec_star
from owl2vec_star import cli


class TestOwl2vec_star(unittest.TestCase):
    """Tests for `owl2vec_star` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_main_function(self):
        """Test main call"""
        owl2vec_star.extract_owl2vec_model("../case_studies/pizza/pizza.owl", "../default.cfg", True, True, True)
	
        

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.standalone)
        #assert result.exit_code == 0
        #assert 'owl2vec_star.cli.standalone' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
        
if __name__ == '__main__':
	unittest.main()
