from setuptools import setup

def readme_file_contents():
    with open('README.rst') as readme_file:
        data = readme_file.read()
    return data

setup(
   name='pvtm',
   version='1.0.0',
   description='Topic Modeling with doc2vec and Gaussian mixture clustering',
   long_description=readme_file_contents(),
   author='David Lenz',
   author_email='david.lenz@wi.jlug.de',
   licence='MIT',
   packages=['pvtm'],  #same as name
   #package_data={'dlutils': ['stopwords.txt']},
   #include_package_data=True,
   #install_requires=['smtplib', 'email'], #external packages as dependencies
)