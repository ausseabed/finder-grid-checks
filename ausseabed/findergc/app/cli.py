#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Command line tool that executes quality assurance checks on grid data
    derived from multibeam echosounder data.
"""

import click
import json
import os

from osgeo import gdal
from typing import Optional, Dict, List, Any

from ausseabed.mbesgc.app.cli import inputs_from_qajson
from ausseabed.mbesgc.lib.data import get_input_details, \
    inputs_from_qajson_checks
from ausseabed.mbesgc.lib.executor import Executor
from ausseabed.mbesgc.lib.check_utils import get_all_check_ids
from ausseabed.findergc.lib.allchecks import all_checks
from ausseabed.qajson.parser import QajsonParser
from ausseabed.qajson.model import QajsonCheck


@click.command()
@click.option(
    '-i', '--input',
    required=False,
    help='Path to input QA JSON file')
@click.option(
    '-gf', '--grid-file',
    required=False,
    help='Path to input grid file (.tif, .bag)')
def cli(
        input,
        grid_file):
    '''Run quality assurance check over input grid file'''

    exe = None

    if grid_file is not None:
        if not os.path.isfile(grid_file):
            click.echo(
                "Grid file ({}) does not exist".format(grid_file),
                err=True)
            sys.exit(os.EX_NOINPUT)

        all_check_ids = get_all_check_ids(all_checks)
        inputs = get_input_details([grid_file])
        for input in inputs:
            input.check_ids_and_params = [(cid, None) for cid in all_check_ids]

        exe = Executor(inputs, all_checks)

    elif input is not None:
        if not os.path.isfile(input):
            click.echo(
                "Input file ({}) does not exist".format(input),
                err=True)
            sys.exit(os.EX_NOINPUT)
        qajson_folder = os.path.dirname(input)
        with open(input) as jsonfile:
            qajson = json.load(jsonfile)
            output = qajson
            spdatachecks = qajson['qa']['survey_products']['checks']
            inputs = inputs_from_qajson(spdatachecks, qajson_folder)

            exe = Executor(inputs, all_checks)
    else:
        click.echo(
            "'-input' or '--grid-file' command line arg must be provided")

    def print_prog(progress):
        click.echo(f"progress = {progress}")
    exe.run(print_prog)

    for check_id, check in exe.check_result_cache.items():
        print()
        print(check_id)
        output = check.get_outputs()
        output_dict = output.to_dict()
        print(json.dumps(output_dict, indent=4))


if __name__ == '__main__':
    cli()
