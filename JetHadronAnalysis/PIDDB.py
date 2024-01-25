def getParticleFractionByMomentum(analysisType, region, particleType, dbCursor):

    # define the SQL query
    query = f"SELECT momentum_bin, {particleType.name.lower()}_fraction, {particleType.name.lower()}_fraction_error FROM particle_fractions WHERE analysis_type = ? AND region = ? ORDER BY momentum_bin;"
    # execute the query
    result = dbCursor.execute(query, (f"{analysisType.name}", f"{region.name}"))
    # get the result
    result = result.fetchall()
    # convert the result to a list
    # return the result
    return result

def getParticleFractionForMomentumBin(analysisType, region, momentum_bin, particleType, dbCursor):

    # define the SQL query
    query = f"SELECT {particleType.name.lower()}_fraction, {particleType.name.lower()}_fraction_error, {particleType.name.lower()}_pid_fit_shape_sys_err, {particleType.name.lower()}_pid_fit_yield_sys_err FROM particle_fractions WHERE analysis_type = ? AND region = ? AND momentum_bin = ?;"
    # execute the query
    result = dbCursor.execute(query, (f"{analysisType.name}", f"{region.name}", f"{momentum_bin.value}"))
    # get the result
    result = result.fetchall()
    # convert the result to a list
    # return the result
    return result

def getParameterByMomentum(analysisType, region, parameter, dbCursor):
    
        # define the SQL query
        query = f"SELECT momentum_bin, {parameter}, {parameter}_error FROM fit_parameters WHERE analysis_type = ? AND region = ? ORDER BY momentum_bin;"
        # execute the query
        result = dbCursor.execute(query, (f"{analysisType.name}", f"{region.name}"))
        # get the result
        result = result.fetchall()
        # convert the result to a list
        # return the result
        return result