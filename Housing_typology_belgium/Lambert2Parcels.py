
#!/usr/bin/env python
# coding: utf-8
import geopandas as gpd
import pandas as pd
import time


def run_time(start):
    end = time.time() - start
    day = end // (24 * 3600)
    end = end % (24 * 3600)
    hour = end // 3600
    end %= 3600
    minutes = end // 60
    end %= 60
    seconds = end
    print("d:h:m:s-> %d:%d:%d:%d" % (day, hour, minutes, seconds))


if __name__ == '__main__':
    
    start = time.time()
    
    Bpn_CaPa_file = "./data/Toute_la_Belgique/Bpn_CaPa.shp"
    lambert_file = "./data/lambert.txt"
    
    #Bpn_CaBu_file = sys.argv[1]    # building shape files
    #Bpn_CaPa_file = sys.argv[2]    # Block shape file

    Bpn_CaPa = gpd.read_file(Bpn_CaPa_file)
    Bpn_CaPa.dropna(subset=['geometry'], inplace = True)
    
    lambert_df = pd.read_csv(lambert_file, sep="\t")
    lambert_gdf = gpd.GeoDataFrame(lambert_df, geometry=gpd.points_from_xy(lambert_df.LambertX, lambert_df.LambertY))
    lambert_gdf.crs = Bpn_CaPa.crs
    
    join_lam2parcel = gpd.sjoin(lambert_gdf, Bpn_CaPa, how="left", op="within")
    Lambert2parcel = pd.DataFrame(join_lam2parcel.head(len(join_lam2parcel)))
    Lambert2parcel.drop(columns=['geometry'], inplace=True)
    
    lam2par_output = "./data/Lamberts2Parcels.csv"
    Lambert2parcel.to_csv(lam2par_output, index=False)
       
    print(run_time(start))
