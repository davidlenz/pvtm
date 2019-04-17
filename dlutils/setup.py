from setuptools import setup

setup(
   name='dlutils',
   version='1.0',
   description='Some Functions that i use over and over again',
   author='David Lenz',
   author_email='david.lenz@wi.jlug.de',
   packages=['dlutils'],  #same as name
   package_data={'dlutils': ['stopwords.txt']},
   include_package_data=True,
#    install_requires=['smtplib', 'email'], #external packages as dependencies
)