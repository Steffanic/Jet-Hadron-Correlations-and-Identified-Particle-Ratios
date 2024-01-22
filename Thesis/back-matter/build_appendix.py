
analysis_type_names = ["PP", "CENTRAL", "SEMICENTRAL"]
assoc_pt_folder_names = ["PT_1_15", "PT_15_2", "PT_2_3", "PT_3_4", "PT_4_5", "PT_5_6", "PT_6_10"]
regions = ["INCLUSIVE", "NEAR_SIDE_SIGNAL", "AWAY_SIDE_SIGNAL", "BACKGROUND"]
particle_types = ["Inclusive", "Pion", "Proton", "Kaon"]

def build_appendix_entry(analysis_type, assoc_pt_folder):
    return f"""
            \\subsection{{{analysis_type} {"-".join(assoc_pt_folder.split("_"))}}}
            \\begin{{figure}}[H]
                \\title{{Region Inclusive}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.INCLUSIVE_Inclusive.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} INCLUSIVE region for Inclusive particles.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_INCLUSIVE_Inclusive}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.INCLUSIVE_Pion.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} INCLUSIVE region for Pions.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_INCLUSIVE_Pion}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.INCLUSIVE_Proton.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} INCLUSIVE region for Protons.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_INCLUSIVE_Proton}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.INCLUSIVE_Kaon.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} INCLUSIVE region for Kaons.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_INCLUSIVE_Kaon}}
                \\end{{subfigure}}
                \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} INCLUSIVE region.}}
                \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_INCLUSIVE}}
            \\end{{figure}}
            \\begin{{figure}}[H]
                \\title{{Region Near-side}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.NEAR_SIDE_SIGNAL_Inclusive.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} NEAR-SIDE region for Inclusive particles.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_NEAR_SIDE_SIGNAL_Inclusive}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.NEAR_SIDE_SIGNAL_Pion.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} NEAR-SIDE region for Pions.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_NEAR_SIDE_SIGNAL_Pion}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.NEAR_SIDE_SIGNAL_Proton.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} NEAR-SIDE region for Protons.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_NEAR_SIDE_SIGNAL_Proton}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.NEAR_SIDE_SIGNAL_Kaon.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} NEAR-SIDE region for Kaons.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_NEAR_SIDE_SIGNAL_Kaon}}
                \\end{{subfigure}}
                \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} NEAR-SIDE region.}}
                \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_NEAR_SIDE_SIGNAL}}
            \\end{{figure}}
            \\begin{{figure}}[H]
                \\title{{Region Away-side}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.AWAY_SIDE_SIGNAL_Inclusive.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} AWAY-SIDE region for Inclusive particles.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_AWAY_SIDE_SIGNAL_Inclusive}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.AWAY_SIDE_SIGNAL_Pion.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} AWAY-SIDE region for Pions.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_AWAY_SIDE_SIGNAL_Pion}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.AWAY_SIDE_SIGNAL_Proton.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} AWAY-SIDE region for Protons.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_AWAY_SIDE_SIGNAL_Proton}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.AWAY_SIDE_SIGNAL_Kaon.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} AWAY-SIDE region for Kaons.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_AWAY_SIDE_SIGNAL_Kaon}}
                \\end{{subfigure}}
                \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} AWAY-SIDE region.}}
                \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_AWAY_SIDE_SIGNAL}}
            \\end{{figure}}
            \\begin{{figure}}[H]
                \\title{{Region Background}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.BACKGROUND_Inclusive.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} BACKGROUND region for Inclusive particles.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_BACKGROUND_Inclusive}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.BACKGROUND_Pion.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} BACKGROUND region for Pions.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_BACKGROUND_Pion}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.BACKGROUND_Proton.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} BACKGROUND region for Protons.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_BACKGROUND_Proton}}
                \\end{{subfigure}}
                \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                    \\centering
                    \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/{assoc_pt_folder}/TPCnSigmaFits/TPCnSigmaFit_Region.BACKGROUND_Kaon.png}}
                    \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} BACKGROUND region for Kaons.}}
                    \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_BACKGROUND_Kaon}}
                \\end{{subfigure}}
                \\caption{{TPC n$\\sigma$ fits for {analysis_type} {"-".join(assoc_pt_folder.split("_"))} BACKGROUND region.}}
                \\label{{fig:appendix_{analysis_type}_{"-".join(assoc_pt_folder.split("_"))}_BACKGROUND}}
            \\end{{figure}}
            \\clearpage
            
    """

def build_yields_and_ratios(analysis_type):
    return f'''
                \\subsection{{{analysis_type} Yields and Ratios}}
                \\begin{{figure}}[H]
                    \\title{{Region Inclusive}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.INCLUSIVE_yields.png}}
                        \\caption{{Particle yields for {analysis_type} INCLUSIVE region.}}
                        \\label{{fig:appendix_{analysis_type}_INCLUSIVE_Inclusive_Yields}}
                    \\end{{subfigure}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.INCLUSIVE_background_subtracted_yields.png}}
                        \\caption{{Particle yields for {analysis_type} INCLUSIVE region with background subtracted.}}
                        \\label{{fig:appendix_{analysis_type}_INCLUSIVE_Inclusive_Yields_Background_Subtracted}}
                    \\end{{subfigure}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.INCLUSIVE_proton_to_pion_ratio.png}}
                        \\caption{{Proton to Pion ratio for {analysis_type} INCLUSIVE region.}}
                        \\label{{fig:appendix_{analysis_type}_INCLUSIVE_Proton_to_Pion_Ratio}}
                    \\end{{subfigure}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.INCLUSIVE_kaon_to_pion_ratio.png}}
                        \\caption{{Kaon to Pion ratio for {analysis_type} INCLUSIVE region.}}
                        \\label{{fig:appendix_{analysis_type}_INCLUSIVE_Kaon_to_Pion_Ratio}}
                    \\end{{subfigure}}
                    \\caption{{Particle yields and ratios for {analysis_type} INCLUSIVE region.}}
                    \\label{{fig:appendix_{analysis_type}_INCLUSIVE_Inclusive_Yields_and_Ratios}}
                \\end{{figure}}
                \\begin{{figure}}[H]
                    \\title{{Region Near-side}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.NEAR_SIDE_SIGNAL_yields.png}}
                        \\caption{{Particle yields for {analysis_type} NEAR-SIDE region.}}
                        \\label{{fig:appendix_{analysis_type}_NEAR_SIDE_SIGNAL_Inclusive_Yields}}
                    \\end{{subfigure}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.NEAR_SIDE_SIGNAL_background_subtracted_yields.png}}
                        \\caption{{Particle yields for {analysis_type} NEAR-SIDE region with background subtracted.}}
                        \\label{{fig:appendix_{analysis_type}_NEAR_SIDE_SIGNAL_Inclusive_Yields_Background_Subtracted}}
                    \\end{{subfigure}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.NEAR_SIDE_SIGNAL_proton_to_pion_ratio.png}}
                        \\caption{{Proton to Pion ratio for {analysis_type} NEAR-SIDE region.}}
                        \\label{{fig:appendix_{analysis_type}_NEAR_SIDE_SIGNAL_Proton_to_Pion_Ratio}}
                    \\end{{subfigure}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.NEAR_SIDE_SIGNAL_kaon_to_pion_ratio.png}}
                        \\caption{{Kaon to Pion ratio for {analysis_type} NEAR-SIDE region.}}
                        \\label{{fig:appendix_{analysis_type}_NEAR_SIDE_SIGNAL_Kaon_to_Pion_Ratio}}
                    \\end{{subfigure}}
                    \\caption{{Particle yields and ratios for {analysis_type} NEAR-SIDE region.}}
                    \\label{{fig:appendix_{analysis_type}_NEAR_SIDE_SIGNAL_Inclusive_Yields_and_Ratios}}
                \\end{{figure}}
                \\begin{{figure}}[H]
                    \\title{{Region Away-side}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.AWAY_SIDE_SIGNAL_yields.png}}
                        \\caption{{Particle yields for {analysis_type} AWAY-SIDE region.}}
                        \\label{{fig:appendix_{analysis_type}_AWAY_SIDE_SIGNAL_Inclusive_Yields}}
                    \\end{{subfigure}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.AWAY_SIDE_SIGNAL_background_subtracted_yields.png}}
                        \\caption{{Particle yields for {analysis_type} AWAY-SIDE region with background subtracted.}}
                        \\label{{fig:appendix_{analysis_type}_AWAY_SIDE_SIGNAL_Inclusive_Yields_Background_Subtracted}}
                    \\end{{subfigure}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.AWAY_SIDE_SIGNAL_proton_to_pion_ratio.png}}
                        \\caption{{Proton to Pion ratio for {analysis_type} AWAY-SIDE region.}}
                        \\label{{fig:appendix_{analysis_type}_AWAY_SIDE_SIGNAL_Proton_to_Pion_Ratio}}
                    \\end{{subfigure}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.AWAY_SIDE_SIGNAL_kaon_to_pion_ratio.png}}
                        \\caption{{Kaon to Pion ratio for {analysis_type} AWAY-SIDE region.}}
                        \\label{{fig:appendix_{analysis_type}_AWAY_SIDE_SIGNAL_Kaon_to_Pion_Ratio}}
                    \\end{{subfigure}}
                    \\caption{{Particle yields and ratios for {analysis_type} AWAY-SIDE region.}}
                    \\label{{fig:appendix_{analysis_type}_AWAY_SIDE_SIGNAL_Inclusive_Yields_and_Ratios}}
                \\end{{figure}}
                \\begin{{figure}}[H]
                    \\title{{Region Background}}
                    \\begin{{subfigure}}[b]{{0.5\\textwidth}}
                        \\centering
                        \\includegraphics[width=\\textwidth]{{figures/png/appendix_plots/{analysis_type}/Region.BACKGROUND_yields.png}}
                        \\caption{{Particle yields for {analysis_type} BACKGROUND region.}}
                        \\label{{fig:appendix_{analysis_type}_BACKGROUND_Inclusive_Yields}}
                    \\end{{subfigure}}
                    \\caption{{Particle yields for {analysis_type} BACKGROUND region.}}
                    \\label{{fig:appendix_{analysis_type}_BACKGROUND_Inclusive_Yields}}
                \\end{{figure}}


    '''


def build_appendix():
    appendix = ""
    appendix += "\\chapter{Appendix}\n"
    for analysis_type in analysis_type_names:
        appendix += f"""
        \\section{{{analysis_type}}}
        """
        appendix += build_yields_and_ratios(analysis_type)
        for assoc_pt_folder in assoc_pt_folder_names:
            appendix += build_appendix_entry(analysis_type, assoc_pt_folder)
    return appendix

if __name__ == "__main__":
    # output to appendix-1.tex
    with open("appendix-1.tex", "w") as f:
        f.write(build_appendix())