"""
Created 18. March 2022 by Daniel Van Opdenbosch, Technical University of Munich

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. It is distributed without any warranty or implied warranty of merchantability or fitness for a particular purpose. See the GNU general public license for more details: <http://www.gnu.org/licenses/>
"""

import glob
import os
import re

outfiles=glob.glob('**/*.out',recursive=True)
os.system('mv read.out read.alt')
os.system('touch read.out')

for i in outfiles:
	filename=os.path.splitext(i)[0]
	name=filename.split('/')[-1]
	j=open(i,'r')
	CF=re.findall('Run *\d* CF:    .\d.\d\d\d',j.read())
	j=open(i,'r')
	VolperAt=re.findall('Volume per atom calculated by starting model = ?.\d.\d\d\d',j.read())
	f=open('read.out','a')
	f.write(' '.join((str(name),str(CF),str(VolperAt)))+'\n')

