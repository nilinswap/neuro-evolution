#include<stdio.h>
#include<stdlib.h>
void reverse(int a[],int start,int end)
{
	while(start<end)
	{
		int temp=a[start];
		a[start]=a[end];
		a[end]=temp;
		start++;
		end--;
		
	}
}
void merge(int a[],int low,int mid,int high)
{
	int i=low;
	int j=mid+1;
	while(i<=mid && a[i]<0)
		i++;
	while(j<=high && a[j]<0)
		j++;
	reverse(a,i,mid);
	reverse(a,mid+1,j-1);
	reverse(a,i,j-1);
	
}
void mergesort(int a[],int low,int high)
{
	if(low<high)
	{
		int mid=(low+high)/2;
		mergesort(a,low,mid);
		mergesort(a,mid+1,high);
		merge(a,low,mid,high);
	}
}
int main()
{
	int n,i;
	scanf("%d",&n);
	int a[n];
	for(i=0;i<n;i++)
		scanf("%d",&a[i]);
	mergesort(a,0,n-1);
	for(i=0;i<n;i++)
		printf("%d ",a[i]);
	return 0;
}
