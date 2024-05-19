from relativisticpy.workbook.workbook import Workbook

def test_general_schild_derivation_from_spherically_symetric_tensor():
    wb = Workbook()

    wb = (
        Workbook()
    )  # Refresh the cache each time you call the method --- I am getting errors because the state is not refreshing
    res = wb.expr(
    """
            Coordinates := [t, r, theta, phi]

            g_{mu}_{nu} := [ 
                            [-A(r),0,0,0], 
                            [0,B(r),0,0], 
                            [0,0,r**2,0], 
                            [0,0,0,r**2*sin(theta)**2]
                        ]

            #T_{mu nu} := 0 # We are looking for a solution in empty space. 
            #G_{mu nu} := R_{mu nu} - (1/2) * R * g_{mu nu}

            # Now we have defined the metric above, we can call any individual component of the Ricci tensor itself (as it is metric dependent)
            eq0 := Ric_{mu=0 nu=0}
            eq1 := Ric_{mu=1 nu=1}
            eq2 := Ric_{mu=2 nu=2}

            eq5 := (eq0*B(r) + eq1*A(r))*(r*B(r))

            B1 := RHS( dsolve(eq5, B(r)) )

            eq6 := simplify( subs(eq2, B(r), B1) )

            A1 := RHS( dsolve(eq6, A(r)) )

            g_{mu}_{nu} := [
                            [A1,0,0,0], 
                            [0,1/A1,0,0], 
                            [0,0,r**2,0], 
                            [0,0,0,r**2*sin(theta)**2]
                        ]

            g_{mu}_{nu} 
    """
    )
    assert str(res.components) == '[[C1 + C2/r, 0, 0, 0], [0, 1/(C1 + C2/r), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]]'

def test_schild_boundary_condition_calculation():
    wb = (
        Workbook()
    )  # Refresh the cache each time you call the method --- I am getting errors because the state is not refreshing
    res = wb.expr(
    """

    Coordinates := [t, r, theta, phi]

    g_{mu}_{nu} := [[1 + C/r, 0, 0, 0], [0, 1/(1 + C/r), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]]

    Gamma^{a}_{c f} := (1/2)*g^{a b}*(d_{c}*g_{b f} + d_{f}*g_{b c} - d_{b}*g_{c f})
    a1 := Gamma^{a:1}_{c:0 f:0}
    exopr := solve( \lim_{r -> oo} a1*r**2 + G*M , C)
    exopr
    """
    )
    assert str(res[0]) == '-2*G*M' # TODO: solve returns a list => Need to be added to sematic analyzer + UI functionality for user to be able to manipulate lists.

def test_k_scalar_derivation_from_schild():
    wb = (
        Workbook()
    )  # Refresh the cache each time you call the method --- I am getting errors because the state is not refreshing
    res = wb.expr(
    """
    Coordinates := [t, r, theta, phi]

    g_{mu}_{nu} := [
                        [-(1 - (2 * G * M) / (c**2*r)), 0, 0, 0],
                        [0, 1 / (1 - (2 * G * M) / (c**2*r)), 0, 0],
                        [0, 0, r**2, 0],
                        [0, 0, 0, r**2 * sin(theta) ** 2]
                    ]

    Gamma^{a}_{c f} := (1/2)*g^{a b}*(d_{c}*g_{b f} + d_{f}*g_{b c} - d_{b}*g_{c f})

    Riemann^{a}_{m b n} := d_{b}*Gamma^{a}_{n m} + Gamma^{a}_{b l}*Gamma^{l}_{n m} - d_{n}*Gamma^{a}_{b m} - Gamma^{a}_{n l}*Gamma^{l}_{b m}

    Ricci_{m}_{n} := Riemann^{a}_{m}_{a}_{n}

    TempOne^{a}^{f}^{h}^{i} := g^{i}^{q}*( g^{h}^{c}*( g^{f}^{b} * Riemann^{a}_{b}_{c}_{q} ) )

    TempTwo_{a}_{f}_{h}_{i} := g_{a}_{n}*Riemann^{n}_{f}_{h}_{i}

    S := TempOne^{a}^{f}^{h}^{i}*TempTwo_{a}_{f}_{h}_{i}

    simplify(S)
    """
    )
    assert str(res) == '48.0*G**2*M**2/(c**4*r**6)'