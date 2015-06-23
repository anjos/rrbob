#!/usr/bin/env python
# encoding: utf-8
# Andre Anjos <andre.anjos@idiap.ch>
# Sat 20 Jun 2015 08:58:03 CEST

from setuptools import setup, find_packages

setup(

   name="rrbob",
   version="0.1",
   description="Basic example of a Reproducible Research Project in Python",
   url='http://github.com/anjos/rrbob',
   license="BSD",
   author='Andre Anjos',
   author_email='andre.anjos@idiap.ch',
   long_description=open('README.rst').read(),

   packages=find_packages(),
   include_package_data=True,
   zip_safe=False,

   install_requires=[
     'setuptools', #always required
     'numpy',
     'bob.db.iris',
     'bob.learn.linear',
     ],

   entry_points = {
     'console_scripts': [
       'paper.py = rr.paper:main',
       ],
     },

   classifiers = [
     'Framework :: Bob',
     'Development Status :: 4 - Beta',
     'Intended Audience :: Developers',
     'License :: OSI Approved :: BSD License',
     'Natural Language :: English',
     'Programming Language :: Python',
     'Programming Language :: Python :: 3',
     'Topic :: Software Development :: Libraries :: Python Modules',
     ],

   )
