import numpy
import pandas
import os
import glob
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

filenamelist,phaselist,XrayDensity_collect,lata_collect,latb_collect,latc_collect,GrainSize100_collect,GrainSize010_collect,GrainSize001_collect,MicroStrain100_collect,MicroStrain010_collect,MicroStrain001_collect,Textur100_collect,Textur010_collect,Textur001_collect,TDS100_collect,TDS010_collect,TDS001_collect,Gewicht_collect,fc_collect,k_collect,J_collect=pickle.load(open('alle.pic','rb'))

Textsum=numpy.array(Textur100_collect)+numpy.array(Textur010_collect)+numpy.array(Textur001_collect)
Textur100_collect/=Textsum
Textur010_collect/=Textsum
Textur001_collect/=Textsum

names=['XrayDensity_collect','lata_collect','latb_collect','latc_collect','GrainSize100_collect','GrainSize010_collect','GrainSize001_collect','MicroStrain100_collect','MicroStrain010_collect','MicroStrain001_collect','Textur100_collect','Textur010_collect','Textur001_collect','TDS100_collect','TDS010_collect','TDS001_collect','Gewicht_collect','fc_collect','k_collect','J_collect']

data=[XrayDensity_collect,lata_collect,latb_collect,latc_collect,GrainSize100_collect,GrainSize010_collect,GrainSize001_collect,MicroStrain100_collect,MicroStrain010_collect,MicroStrain001_collect,Textur100_collect,Textur010_collect,Textur001_collect,TDS100_collect,TDS010_collect,TDS001_collect,Gewicht_collect,fc_collect,k_collect,J_collect]

plt.clf()
fig,ax1=plt.subplots()
dataframe=pandas.DataFrame(numpy.array(data).transpose(),columns=names)
corr=dataframe.corr()
cax=ax1.matshow(corr,cmap='coolwarm',vmin=-1,vmax=1,interpolation='none')
cbar=fig.colorbar(cax)
ticks=numpy.arange(0,len(dataframe.columns),1)
ax1.set_xticks(ticks)
ax1.set_yticks(ticks)
ax1.set_xticklabels(dataframe.columns)
ax1.set_yticklabels(dataframe.columns)
plt.xticks(rotation=90)
ax1.set_ylim([len(dataframe.columns)-0.5,-0.5])
plt.tight_layout(pad=0.1)
plt.savefig('corr.png')

phases=[]
for p,valuep in enumerate(phaselist):
	phases.append(filenamelist[p]+':'+valuep)

argsort=numpy.argsort(filenamelist)

for i,valuei in enumerate(data):
	plt.clf()
	yerr=numpy.array([])
	for j in valuei:
		if '+/-' in str(j):
			yerr=numpy.append(yerr,j.uncertainty)
		else:
			yerr=numpy.append(yerr,0)
	plt.errorbar(numpy.array(phases)[argsort],numpy.array(valuei)[argsort],yerr=yerr[argsort])
	plt.xticks(rotation=90)
	plt.tight_layout(pad=0.1)
	plt.savefig(names[i]+'.png')

