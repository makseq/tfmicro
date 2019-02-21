"""
TFMicro
Copyright (C) 2018 Maxim Tkachenko

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import callbacks
import data
import model
import model_loader
import model_predictor
import threadgen
import workers

from .version import get_git_version, get_short_version

__author__ = 'Max Tkachenko'
__email__ = 'makseq@gmail.com'
__version__ = get_short_version()
__git_version__ = get_git_version()
__description__ = 'Framework for Tensorflow models'
